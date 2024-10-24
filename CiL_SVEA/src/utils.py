import torch
import numpy as np
import os
import glob
import json
import random
import augmentations
import subprocess
from datetime import datetime


class eval_mode(object):
	def __init__(self, *models):
		self.models = models

	def __enter__(self):
		self.prev_states = []
		for model in self.models:
			self.prev_states.append(model.training)
			model.train(False)

	def __exit__(self, *args):
		for model, state in zip(self.models, self.prev_states):
			model.train(state)
		return False


def soft_update_params(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(
			tau * param.data + (1 - tau) * target_param.data
		)


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def write_info(args, fp):
	data = {
		'timestamp': str(datetime.now()),
		'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
		'args': vars(args)
	}
	with open(fp, 'w') as f:
		json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
	path = os.path.join('setup', 'config.cfg')
	with open(path) as f:
		data = json.load(f)
	if key is not None:
		return data[key]
	return data


def make_dir(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
	fpath = os.path.join(dir_path, f'*.{filetype}')
	fpaths = glob.glob(fpath, recursive=True)
	if sort:
		return sorted(fpaths)
	return fpaths


def prefill_memory(obses, capacity, obs_shape):
	c,h,w = obs_shape
	for _ in range(capacity):
		frame = np.ones((3,h,w), dtype=np.uint8)
		obses.append(frame)
	return obses


def prefill_memory2(obses, capacity, obs_shape):
	nump,e,h,w = obs_shape
	for _ in range(capacity):
		frame = np.ones((nump,1,h,w), dtype=np.uint8)
		obses.append(frame)
	return obses

class ReplayBuffer(object):
	"""Buffer to store environment transitions"""
	def __init__(self, device, obs_shape, action_shape, Vh_shape, capacity, batch_size, crop, prefill=True):
		self.capacity = capacity
		self.device = device
		self.batch_size = batch_size
		if crop == 'vitcrop':
			self.crop = augmentations.vit_crop
		else:
			self.crop = lambda x: x

		self._obses = []
		self._Vhs = []
		self._Vhs_next = []
		if prefill:
			self._obses = prefill_memory(self._obses, capacity, obs_shape)
			self._Vhs = prefill_memory2(self._Vhs, capacity, Vh_shape)
			self._Vhs_next = prefill_memory2(self._Vhs, capacity, Vh_shape)
		self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
		self.rewards = np.empty((capacity, 1), dtype=np.float32)
		self.not_dones = np.empty((capacity, 1), dtype=np.float32)

		self.idx = 0
		self.full = False

	def add(self, obs, action, reward, next_obs, Vh, Vh_next, done):
		obses = (obs, next_obs)
		if self.idx >= len(self._obses):
			self._obses.append(obses)
			self._Vhs.append(Vh)
			self._Vhs_next.append(Vh_next)
		else:
			self._obses[self.idx] = (obses)
			self._Vhs[self.idx] = (Vh)
			self._Vhs_next[self.idx] = (Vh_next)
		np.copyto(self.actions[self.idx], action)
		np.copyto(self.rewards[self.idx], reward)
		np.copyto(self.not_dones[self.idx], not done)

		self.idx = (self.idx + 1) % self.capacity
		self.full = self.full or self.idx == 0

	def _get_idxs(self, n=None):
		if n is None:
			n = self.batch_size
		return np.random.randint(
			0, self.capacity if self.full else self.idx, size=n
		)

	def _encode_obses(self, idxs):
		obses, next_obses , Vhs ,Vhs_next= [], [], [] ,[]
		for i in idxs:
			obs, next_obs = self._obses[i]
			Vh = self._Vhs[i]
			Vh_next = self._Vhs_next[i]
			obses.append(np.array(obs, copy=False))
			next_obses.append(np.array(next_obs, copy=False))
			Vhs.append(np.array(Vh, copy=False))
			Vhs_next.append(np.array(Vh_next, copy=False))
		return np.array(obses), np.array(next_obses), np.array(Vhs), np.array(Vhs_next)

	def sample(self, n=None):
		idxs = self._get_idxs(n)

		obs, next_obs, Vhs, Vhs_next  = self._encode_obses(idxs)
		Vhs = torch.as_tensor(Vhs).to(self.device).float()
		Vhs_next = torch.as_tensor(Vhs_next).to(self.device).float()
		obs = torch.as_tensor(obs).to(self.device).float()
		next_obs = torch.as_tensor(next_obs).to(self.device).float()
		actions = torch.as_tensor(self.actions[idxs]).to(self.device)
		rewards = torch.as_tensor(self.rewards[idxs]).to(self.device)
		not_dones = torch.as_tensor(self.not_dones[idxs]).to(self.device)

		obs = self.crop(obs)
		next_obs = self.crop(next_obs)

		return obs, actions, rewards, next_obs, Vhs, Vhs_next, not_dones


class LazyFrames(object):
	def __init__(self, frames, extremely_lazy=True):
		self._frames = frames
		self._extremely_lazy = extremely_lazy
		self._out = None

	@property
	def frames(self):
		return self._frames

	def _force(self):
		if self._extremely_lazy:
			return np.concatenate(self._frames, axis=0)
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=0)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		if self._extremely_lazy:
			return len(self._frames)
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

	def count(self):
		if self.extremely_lazy:
			return len(self._frames)
		frames = self._force()
		return frames.shape[0]//3

	def frame(self, i):
		return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
	"""Returns number of params in network"""
	count = sum(p.numel() for p in net.parameters())
	if as_int:
		return count
	return f'{count:,}'
