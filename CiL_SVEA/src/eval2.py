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


def get_camera_params( scale,scale2=0, dynamic=False):
  return dict(
      vertical_delta=np.pi / 2 * scale2,
      horizontal_delta=np.pi / 2 * scale2,
      # Limit camera to -90 / 90 degree rolls.
      roll_delta=np.pi / 2. * scale,
      vel_std=.1 * scale2 if dynamic else 0.,
      max_vel=.4 * scale2 if dynamic else 0.,
      roll_std=np.pi / 300 * scale2 if dynamic else 0.,
      max_roll_vel=np.pi / 50 * scale2 if dynamic else 0.,
      # Allow the camera to zoom in at most 50%.
      max_zoom_in_percent=.5 * scale2,
      # Allow the camera to zoom out at most 200%.
      max_zoom_out_percent=1.5 * scale2,
      limit_to_upper_quadrant= False, #'reacher' not in domain_name,
  )



def distraction_wrap(env,scale):
  camera_kwargs = get_camera_params(scale=scale, dynamic=False)#, dynamic=True)
  return DistractingCameraEnv(env, camera_id=0, **camera_kwargs)


def evaluate(env, agent, video, num_episodes, eval_mode, adapt=False):
	episode_rewards = []
    # Adjust camera setting



	for i in tqdm(range(num_episodes)):
		if adapt:
			ep_agent = deepcopy(agent)
			ep_agent.init_pad_optimizer()
		else:
			ep_agent = agent
		obs = env.reset()
		video.init(enabled=True)
		done = False
		episode_reward = 0
		count = 0
		while not done:
			count +=1
			with utils.eval_mode(ep_agent):
				action = ep_agent.select_action(obs)
			next_obs, reward, done, _ = env.step(action)
			video.record(env, eval_mode)
			episode_reward += reward
			if adapt:
				ep_agent.update_inverse_dynamics(*augmentations.prepare_pad_batch(obs, next_obs, action))
			obs = next_obs
		print("(np.array(obs)", np.array(obs).shape, "count ", count)
		#print(np.array(obs))
		video.save(f'eval_{eval_mode}_{i}.mp4')
		episode_rewards.append(episode_reward)

	return np.mean(episode_rewards), np.std(episode_rewards)

from env.wrappers import FrameStack
import dmc2gym 

def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		mode=args.eval_mode
	)

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
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)
	print(model_dir)
	agent = torch.load(os.path.join(model_dir, str(args.train_steps)+'.pt'))
	agent.train(False)

	#for scale in  np.arange(0.0, 0.2, 0.01):  
	for scale in  np.arange(0.0, 1.0, 0.1):  
		gym.logger.set_level(40)
		env = make_env2(
			domain_name=args.domain_name,
			task_name=args.task_name,
			seed=args.seed+42,
			episode_length=args.episode_length,
			action_repeat=args.action_repeat,
			mode=args.eval_mode,
			#scale= scale,
		)
		env = dmc2gym.make(domain_name="walker", task_name="walk",  height=100,
                width=100,frame_skip=args.action_repeat, visualize_reward=False, from_pixels=True)#frame_skip=1,
		env = distraction_wrap(env,scale=scale)
		env = FrameStack(env,3) 
        
		#time_step = env.reset()
		#distraction_wrap(env,scale=scale)
		#env =env.reset()

		print(f'\nEvaluating {work_dir} for {args.eval_episodes} episodes (mode: {args.eval_mode})')
		reward,std = evaluate(env, agent, video, 25, args.eval_mode)#args.eval_episodes
		print('Scale', scale, ' - Reward:', int(reward) ,' +/- ', int(std))

	adapt_reward = None
	if args.algorithm == 'pad':
		env = make_env(
			domain_name=args.domain_name,
			task_name=args.task_name,
			seed=args.seed+42,
			episode_length=args.episode_length,
			action_repeat=args.action_repeat,
			mode=args.eval_mode
		)
		adapt_reward = evaluate(env, agent, video, args.eval_episodes, args.eval_mode,args.camera_id, adapt=True)
		print('Adapt reward:', int(adapt_reward))

	# Save results
#	torch.save({
#		'args': args,
#		'reward': reward,
#		'adapt_reward': adapt_reward
#	}, results_fp)
#	print('Saved results to', results_fp)


if __name__ == '__main__':
	args = parse_args()
	main(args)
