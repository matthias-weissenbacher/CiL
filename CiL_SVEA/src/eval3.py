import torch
import torchvision
import os
import numpy as np
import gym
import utils
from copy import deepcopy
from tqdm import tqdm
from arguments import parse_args
from env.wrappers import make_env, make_env2
from algorithms.factory import make_agent
from video import VideoRecorder
import augmentations



# Usage







class FrameStacker:
    def __init__(self, num_frames, frame_shape=(3, 96, 96)):
        self.num_frames = num_frames
        # Initialize frames with zeros
        self.frames = np.zeros((num_frames, *frame_shape))

    def add_frame(self, frame):
        assert frame.shape == self.frames.shape[1:], "The frame shape does not match the expected shape."
        # Roll frames and replace the last frame
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = frame

    def get_stacked_frames(self):
        # Stack frames along the 0 dimension (assuming frame shape is (3, 96, 96))
        stacked_frames = np.vstack(self.frames)
        return stacked_frames
    
class FrameStacker:
    def __init__(self, num_frames, frame_shape=(3, 96, 96)):
        self.num_frames = num_frames
        # Initialize frames with zeros
        self.frames = np.zeros((num_frames, *frame_shape))

    def add_frame(self, frame):
        assert frame.shape == self.frames.shape[1:], "The frame shape does not match the expected shape."
        # Roll frames and replace the first frame with the new frame
        self.frames = np.roll(self.frames, shift=1, axis=0)
        self.frames[0] = frame

    def get_stacked_frames(self):
        # Stack frames along the 0 dimension (assuming frame shape is (3, 96, 96))
        stacked_frames = np.vstack(self.frames)
        return stacked_frames
    
from camera import DistractingCameraEnv

def get_camera_params( scale, dynamic):
  return dict(
      vertical_delta=np.pi / 2 * scale,
      horizontal_delta=np.pi / 2 * scale,
      # Limit camera to -90 / 90 degree rolls.
      roll_delta=np.pi / 2. * scale,
      vel_std=.1 * scale if dynamic else 0.,
      max_vel=.4 * scale if dynamic else 0.,
      roll_std=np.pi / 300 * scale if dynamic else 0.,
      max_roll_vel=np.pi / 50 * scale if dynamic else 0.,
      # Allow the camera to zoom in at most 50%.
      max_zoom_in_percent=.5 * scale,
      # Allow the camera to zoom out at most 200%.
      max_zoom_out_percent=1.5 * scale,
      limit_to_upper_quadrant= False, #'reacher' not in domain_name,
  )


def distraction_wrap(env,scale):
  camera_kwargs = get_camera_params(scale=scale, dynamic=True)
  return DistractingCameraEnv(env, camera_id=0, **camera_kwargs)


from dm_control import suite
import dmc2gym

def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)

	# Initialize environments
	gym.logger.set_level(40)
	envd = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		mode=args.eval_mode
	)

	env = distraction_wrap(suite.load(domain_name="walker", task_name="walk"),scale=0.1)
	time_step = env.reset()

	# Set working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name+ '_'+ args.run_name , args.algorithm, args.exp_suffix, str(args.seed))
	print('Working directory:', work_dir)
	assert os.path.exists(work_dir), 'specified working directory does not exist'
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

	# Check if evaluation has already been run
	results_fp = os.path.join(work_dir, args.eval_mode+str(args.camera_id)+'.pt')
#	assert not os.path.exists(results_fp), f'{args.eval_mode+stra(args.camera_id)} results already exist for {work_dir}'

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	cropped_obs_shape = (3*args.frame_stack, 84, 84)
	if args.crop in {'none'}:
		image_size = 84
		image_crop_size = 84
	elif args.crop in {'vitcrop'}:
		image_size = 96
		image_crop_size = 96
	cropped_obs_shape = (3*args.frame_stack, image_crop_size, image_crop_size)
	print('Observations:', envd.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=envd.action_space.shape,
		args=args
	)
	print(model_dir)
	agent = torch.load(os.path.join(model_dir, str(args.train_steps)+'.pt'))
	agent.train(False)

	for scale in  np.arange(0.0, 1.0, 0.2):  
		#gym.logger.set_level(40)
		#env = suite.load(domain_name="walker", task_name="walk",visualize_reward=False)
		env = dmc2gym.make(domain_name="walker", task_name="walk", frame_skip=4, visualize_reward=False, from_pixels=True)#frame_skip=1,
		env = distraction_wrap(env,scale=scale)
		time_step = env.reset()
		#time_step = env.reset()
		#distraction_wrap(env,scale=scale)
		#env =env.reset()

		print(f'\nEvaluating {work_dir} for {args.eval_episodes} episodes (mode: {args.eval_mode})')
		reward,std = evaluate(env, agent, video,1 , args.eval_mode)#args.eval_episodes
		print('Scale', scale, ' - Reward:', int(reward) ,' +/- ', int(std))

import numpy as np
from PIL import Image



# Convert numpy array to PIL Image


# Save the image



def evaluate(env, agent, video, num_episodes, eval_mode, adapt=False):
	episode_rewards = []
    # Adjust camera setting

	for i in tqdm(range(num_episodes)):
		frame_stacker = FrameStacker(3, (3, 100, 100)) 
		ep_agent = agent
		obs = env.reset()

		done = False
		episode_reward = 0
		count = 1
		action = np.zeros((6))# np.random.uniform(low=-1, high=1, size=env.action_spec().shape)
		while not done:
			image = env.physics.render(height=100, width=100, camera_id=0).transpose(2,1,0)
			frame_stacker.add_frame(image) #np.array(obs))
			if count % 1 == 0:  # Only choose a new action every `action_repeat` timesteps
				stacked_frames = frame_stacker.get_stacked_frames()
				action = ep_agent.select_action(stacked_frames)
			#action = np.random.uniform(low=-1, high=1, size=env.action_spec().shape)
			next_obs, reward, done, _ = env.step(action)
			episode_reward += reward
			obs = next_obs
			count +=1
			#print("image.shape", image.shape)
		episode_rewards.append(episode_reward)
		#print(image.shape)
		#image = Image.fromarray(np.array(image).transpose(1,2,0))
		#image.save('obs.png')

	return np.mean(episode_rewards), np.std(episode_rewards)




if __name__ == '__main__':
	args = parse_args()
	main(args)
