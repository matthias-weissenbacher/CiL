import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder


def evaluate(env, agent, video, num_episodes, L, step, test_env=False):
	episode_rewards = []
	agent.shared_cnn.is_training = False
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
			obs, reward, done, _ = env.step(action)
			video.record(env)
			episode_reward += reward

		if L is not None:
			_test_env = '_test_env' if test_env else ''
			video.save(f'{step}{_test_env}.mp4')
			L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
		episode_rewards.append(episode_reward)
	agent.shared_cnn.is_training = True
	return np.mean(episode_rewards)


def main(args):
	# Set seed
	mult = 5
	patch_size = 8
	patch_idxs = create_patching_idxs(dim = 96 , ps= mult*patch_size  , ps2 = patch_size ) 
	utils.set_seed_everywhere(args.seed)
	save_img = True
	# Initialize environments
	gym.logger.set_level(40)
	if args.crop in {'none'}:
		image_size = 84
		image_crop_size = 84
	elif args.crop in {'vitcrop'}:
		image_size = 96
		image_crop_size = 96
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=image_size,
		mode=args.train_mode
	)
	test_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=image_size,
		mode=args.eval_mode
	)

	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name+ '_'+ args.run_name , args.algorithm, args.exp_suffix,  str(args.seed))
	print('Working directory:', work_dir)
	assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	cropped_obs_shape = (3*args.frame_stack, image_crop_size, image_crop_size)
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	device =  torch.device("cuda:"+ str(args.device_id))
	replay_buffer = utils.ReplayBuffer(
		device =device,
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		Vh_shape = ((96//8)**2,1,6,6),
		capacity=args.train_steps,
		batch_size=args.batch_size,
		crop=args.crop
	)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)

	start_step, episode, episode_reward, done = 0, 0, 0, True
	L = Logger(work_dir)
	start_time = time.time()
	for step in range(start_step, args.train_steps+1):
		if done:
			if step > start_step:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				print('Evaluating:', work_dir)
				L.log('eval/episode', episode, step)
				evaluate(env, agent, video, args.eval_episodes, L, step)
				evaluate(test_env, agent, video, args.eval_episodes, L, step, test_env=True)
				L.dump(step)

			# Save agent periodically
			if step > start_step and step % args.save_freq == 0:
				torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

			L.log('train/episode_reward', episode_reward, step)

			obs = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1

			L.log('train/episode', episode, step)

		# Sample action for data collection
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with utils.eval_mode(agent):
				with torch.no_grad():
						agent.shared_cnn.is_training = False
						action = agent.sample_action(obs)
						agent.shared_cnn.is_training = True
    
		# Run training update
		if step >= args.init_steps:
			num_updates = args.init_steps if step == args.init_steps else 1
			for _ in range(num_updates):
				agent.update(replay_buffer, L, step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		#print('*** observation shape ***', np.array(obs).shape)
		Vh = get_Vh(obs,patch_idxs)
		Vh_next = get_Vh(obs,patch_idxs,choice_next = True)
		if save_img:
				save_frames_to_images(np.array(obs))
				save_frames_to_images2(np.array(obs))
				save_img = False
		replay_buffer.add(obs, action, reward, next_obs, Vh,Vh_next, done_bool)
		episode_reward += reward
		obs = next_obs

		episode_step += 1

	print('Completed training for', work_dir)
from PIL import Image
import numpy as np
import torch.nn.functional as F

def save_frames_to_images(frames):
    frames = frames.reshape((3,3,96,96))
    # Assuming frames is a numpy array of shape (3, 3, 96, 96)
    for idx, frame in enumerate(frames):
        # Convert the frame to uint8 for saving as an image
        img_frame = (frame * 255).astype(np.uint8)
        img = Image.fromarray(img_frame.transpose(1, 2, 0))  # Convert to HxWxC format for PIL
        img.save(f"frame_{idx}.png")
        
def save_frames_to_images2(frames):
    frames = frames.reshape((3,3,96,96)).transpose(1,0,2,3)
    # Assuming frames is a numpy array of shape (3, 3, 96, 96)
    for idx, frame in enumerate(frames):
        # Convert the frame to uint8 for saving as an image
        img_frame = (frame * 255).astype(np.uint8)
        img = Image.fromarray(img_frame.transpose(1, 2, 0))  # Convert to HxWxC format for PIL
        img.save(f"frame_p_{idx}.png")

        

def unique_svd(x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
           # Vhinv = torch.linalg.pinv(Vh)
          #  det_Vh = torch.det(Vh)

        
        return U,Sv,Vh.to(device)#,det_Vh.to(device)
    
def get_patches_flat( x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
        
def get_patches_mini(x,patch_idxs,patch_size=8,mult=5,num_patches=(96//8)**2):
        bs, h,w, c = x.shape
        patch_size_l = patch_size*mult
        padding = int((patch_size)**(mult-1)//2) # padding = 3*8//2 =3*4 = 4
        y = F.pad(x.permute(0,3,1,2), (padding, padding, padding, padding), mode='constant', value=0)
        y = torch.index_select(y.flatten(-2), 2, patch_idxs.to(x.device))
        #print(y.shape)
        y =y.reshape(bs,c,num_patches ,patch_size_l,patch_size_l).permute(0,2,3,4,1)
        y = get_patches_flat(y.reshape(bs*num_patches ,patch_size_l,patch_size_l,c), ps=patch_size)
        y = y.reshape((bs, num_patches, mult*mult,patch_size*patch_size, c)).mean(2)
        #print(y.shape)
        return y.reshape((bs*num_patches,patch_size*patch_size, c)) #



        
def create_patching_idxs(dim = 64, ps= 3*8, ps2 = 8):
        #ps = self.mult*ps2
        pad_dim = ps2 #int((ps-1)/2) 
        image_dim_pad = (2*pad_dim  + dim)
        arr = torch.tensor(list(range(image_dim_pad**2))).reshape((image_dim_pad,image_dim_pad)).long()
        idxs = torch.zeros((dim//ps2,dim//ps2,ps,ps)).long()
        for i in range(dim//ps2):
            for j in range(dim//ps2):
                    idxs[i,j,:,:] =  arr[i:i +ps ,j:j+ps]

        return idxs.flatten()

def get_Vh(obs,patch_idxs,choice_next = False):
        with torch.no_grad():
                    num_patches = (96//8)**2
                    x = torch.tensor(np.array(obs)).unsqueeze(0).permute(0,2,3,1).float()
                    bs, idim, idim , cs = x.size() 
                    x = x.reshape((bs, idim, idim,cs//3, 3)).permute(0,1,2,4,3)
                    if choice_next:
                        x0 =  get_patches_mini(x[:,:,:,:,1:].flatten(-2),patch_idxs=patch_idxs)
                    else:
                        x0 =  get_patches_mini(x[:,:,:,:,:-1].flatten(-2),patch_idxs=patch_idxs)
                    #print( x0.shape)

                    #x = get_patches_flat(x[:,:,:,:,1:].flatten(-2),ps= patch_size)
                    _,_,Vh = unique_svd(x0)

                   # mask = det_Vh != 0
                    # Expand the mask and determinant to have the same shape as Vh for broadcasting
                  #  expanded_mask = mask[:, None, None].expand_as(Vh)
                   # expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)

                  #  Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]


                    Vh=Vh.reshape(bs,num_patches,1,6,6)
                    
                    
        return Vh
             
             

if __name__ == '__main__':
	args = parse_args()
	main(args)
