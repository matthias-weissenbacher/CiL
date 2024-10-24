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

import math
def evaluate(env, agent,  num_episodes, step, L= None, test_env=False):
	episode_rewards = []
	agent.shared_cnn.is_training = False
    
	for i in range(num_episodes):
		cM = rand_orth() #euler_to_so3(5*torch.rand((1,3)))
		obs = env.reset()
		#video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				obs_n = (torch.tensor(np.array(obs)).permute(1,2,0).reshape(96,96,3,1,3)).float()@cM
				#print(obs_n.shape) #@cM).numpy() 
				action = agent.select_action(np.array(obs_n.flatten(-3).permute(2,0,1)))
			obs, reward, done, _ = env.step(action)
			#video.record(env)
			episode_reward += reward
		print(f'eval/episode_reward', episode_reward, step)
		if L is not None:
			_test_env = '_test_env' if test_env else ''
			video.save(f'{step}{_test_env}.mp4')
			L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
		episode_rewards.append(episode_reward)
	agent.shared_cnn.is_training = True
	return np.mean(episode_rewards)

def rand_orth():
    
    U = torch.tensor([[[ 0.5750+0.0000j,  0.0707-0.5742j,  0.0707+0.5742j],
         [-0.8000+0.0000j, -0.0984-0.4127j, -0.0984+0.4127j],
         [-0.1714+0.0000j,  0.6966+0.0000j,  0.6966-0.0000j]]])
    real =  torch.rand((1,1))
    imag = torch.sqrt(torch.abs(1 -real**2))
    diag = torch.complex(torch.tensor([1.0,real,real]), torch.tensor([0.0,imag,-imag]))
   
    #orth = torch.real(U@torch.diag_embed(diag)@ torch.conj(U.transpose(-2,-1)))
    orth = torch.real(torch.conj(U.transpose(-2,-1))@torch.diag_embed(diag)@U)
    #print(orth)
    #print("check", U@torch.conj(U.transpose(-2,-1)))
        
    return orth
    
def euler_to_so3(param,embed_dim=1):
        # Ensure that param has shape (embed_dim, 3)
        assert param.dim() == 2 and param.size(1) == 3, "Input tensor should have shape (embed_dim, 3)"

        # Create the Z-X-Z rotation matrices
        phi, psi = math.pi + math.pi * torch.tanh(param[:,0]), math.pi + math.pi * torch.tanh(param[:,1])
        theta = math.pi/2 + math.pi/2 * torch.tanh(param[:,2])

        embed_dim = phi.size(0)

        g_phi = torch.zeros((embed_dim, 3, 3)).to(param.device)
        g_phi[:,0,0] = torch.cos(phi)
        g_phi[:,0,1] = -torch.sin(phi)
        g_phi[:,1,0] = torch.sin(phi)
        g_phi[:,1,1] = torch.cos(phi)
        g_phi[:,2,2] = 1

        g_theta = torch.zeros((embed_dim, 3, 3)).to(param.device)
        g_theta[:,0,0] = 1
        g_theta[:,1,1] = torch.cos(theta)
        g_theta[:,1,2] = -torch.sin(theta)
        g_theta[:,2,1] = torch.sin(theta)
        g_theta[:,2,2] = torch.cos(theta)

        g_psi = torch.zeros((embed_dim, 3, 3)).to(param.device)
        g_psi[:,0,0] = torch.cos(psi)
        g_psi[:,0,1] = -torch.sin(psi)
        g_psi[:,1,0] = torch.sin(psi)
        g_psi[:,1,1] = torch.cos(psi)
        g_psi[:,2,2] = 1

        # Multiply the matrices to get the combined rotation
        g = torch.matmul(g_phi, torch.matmul(g_theta, g_psi))

        return g


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
	#assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
	#utils.make_dir(work_dir)
	#model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	#video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	#video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	#utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	cropped_obs_shape = (3*args.frame_stack, image_crop_size, image_crop_size)
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	device =  torch.device("cuda:"+ str(args.device_id))
	
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)
	step = 100000
	checkpointpath = os.path.join(work_dir+'/model', f'{step}.pt')
	agent.eval()
	#agent.to(flags.device)
	#agent.training = False
	#checkpoint = torch.load(checkpointpath, map_location="cpu")
	agent = torch.load(checkpointpath)
	#agent.load_state_dict(checkpoint["model_state_dict"])
	agent.eval()
	print(agent.shared_cnn.weightOrth.data)
			# Evaluate agent periodically

	print('Evaluating:', work_dir)
	#L.log('eval/episode', episode, step)
	#evaluate(env, agent, video, args.eval_episodes, L, step)
	evaluate(test_env, agent, args.eval_episodes, step, test_env=True)
#L.dump(step)


	print('Completed testing for', work_dir)
    
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
