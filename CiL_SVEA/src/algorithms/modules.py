import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import algorithms.vit as vit
import algorithms.Sit as Sit
import algorithms.CSit as CSit
import utils



def _get_out_shape_cuda(in_shape, layers, device):
	x = torch.randn(*in_shape).to(device).unsqueeze(0)
	return layers(x).squeeze(0).shape

def _get_out_shape(in_shape, layers):
	x = torch.randn(*in_shape).unsqueeze(0)
	return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability."""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
	"""Apply squashing function.
	See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
	"""
	mu = torch.tanh(mu)
	if pi is not None:
		pi = torch.tanh(pi)
	if log_pi is not None:
		log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
	return mu, pi, log_pi


def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CenterCrop(nn.Module):
	def __init__(self, size):
		super().__init__()
		assert size in {84, 96, 100, 112}, f'unexpected size: {size}'
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		assert x.size(3) == 100, f'unexpected size: {x.size(3)}'
		if self.size == 96:
			p = 2
		elif self.size == 84:
			p = 8
		return x[:, :, p:-p, p:-p]


class Time2Space(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		x = x.view(x.size(0), x.size(1)//3, 3, x.size(-2), x.size(-1)) # (B, T, C, H, W)
		return torch.cat(torch.unbind(x, dim=1), dim=-1) # (B, C, H, (TW))


class NormalizeImg(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x/255.


class Flatten(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		x = x.reshape(x.size(0), -1)

		return x.view(x.size(0), -1)


class RLProjection(nn.Module):
	def __init__(self, in_shape, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.projection = nn.Sequential(
			nn.Linear(in_shape[0], out_dim),
			nn.LayerNorm(out_dim),
			nn.Tanh()
		)
		self.apply(weight_init)
	
	def forward(self, x):
		return self.projection(x)


class SODAMLP(nn.Module):
	def __init__(self, projection_dim, hidden_dim, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.mlp = nn.Sequential(
			nn.Linear(projection_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, out_dim)
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(x)


class SharedTransformer(nn.Module):
	def __init__(self, obs_shape, device,  patch_size=8, embed_dim=128, depth=4, num_heads=8, mlp_ratio=1., qvk_bias=False):
		super().__init__()
		assert len(obs_shape) == 3
		self.frame_stack = obs_shape[0]//3
		self.img_size = obs_shape[-1]
		self.patch_size = patch_size
		self.embed_dim = embed_dim
		self.depth = depth
		self.num_heads = num_heads
		self.mlp_ratio = mlp_ratio
		self.qvk_bias = qvk_bias

		self.preprocess = nn.Sequential(CenterCrop(size=self.img_size), NormalizeImg())
		#self.transformer = vit.VisionTransformer(
		self.transformer = Sit.Sit(
#		self.transformer = CSit.CSit(
			img_size=self.img_size,
			patch_size=patch_size,
			in_chans=self.frame_stack*3,
			embed_dim=embed_dim,
			depth=depth,
			num_heads=num_heads,
			mlp_ratio=mlp_ratio,
			qkv_bias=qvk_bias,
		).to(device)# .cuda()
		self.out_shape = _get_out_shape_cuda(obs_shape, nn.Sequential(self.preprocess, self.transformer),device)

	def forward(self, x):
		x = self.preprocess(x)
		return self.transformer(x)

import math

class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        self.img_size = obs_shape[-1]
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.embed_dim = 12

        self.layers = [CenterCrop(size=self.img_size), NormalizeImg()]
        #self.layers.append(nn.Conv2d(obs_shape[0], num_filters, 3, stride=2))
        self.layers.append(nn.Conv2d(self.embed_dim, num_filters, 3, stride=2))
        for i in range(1, num_layers):
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers = nn.Sequential(*self.layers)
    #    print("obs_shape",obs_shape)
        obs_shape =  (self.embed_dim,obs_shape[1],obs_shape[2])
        self.out_shape = _get_out_shape(obs_shape, self.layers)
      #  print(self.out_shape )
        self.apply(weight_init)

       
     #   self.proj_input2 = nn.Linear(30,15)
      #  self.proj_input3 = nn.Linear(30,15)

        
         #num_filters -2
        self.embed_dim2 = 1
        self.patch_size = 8     
        self.mult = 5
       # self.proj_input = nn.Linear(6,6)
        #self.proj_inputf = nn.Linear(self.embed_dim2*6 + 6,num_filters//2)
        self.patch_size_l = self.mult*self.patch_size 
        #self.idim = img_size
        self.patch_idxs = self.create_patching_idxs(dim = 96 , ps= self.patch_size_l  , ps2 = self.patch_size ) 
        
        
        in_chans = 9
        stack = in_chans //3
        self.num_patches = (96 // self.patch_size )**2
          
            
        self.weightTime= nn.Parameter(torch.Tensor(self.embed_dim2 +1,(stack-1), (stack-1))) 
        nn.init.kaiming_uniform_(self.weightTime, a=math.sqrt(5))    
        
     #   self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim2//2, 3,3)) 
      #  nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightOrth = nn.Parameter(torch.Tensor(self.embed_dim2, 3))
        nn.init.kaiming_uniform_(self.weightOrth, a=math.sqrt(5))
        
        self.is_training = True
    

    def euler_to_so3(self, param,embed_dim):
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



    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
           # Vhinv = torch.linalg.pinv(Vh)
          #  det_Vh = torch.det(Vh)

        
        return U,Sv,Vh.to(device)#,det_Vh.to(device)
    
    def reconstruct_image(self,patches, ps=8, img_dim = 64):
        xy_dim = img_dim // ps
        
        bs,num_p, num_pn, f = patches.size()    
        patches = patches.reshape(bs, num_p ,ps,ps,f)
        img = patches.reshape((bs,xy_dim,xy_dim,ps, ps,f)).permute(0,1,3,2,4,5)
        img = img.reshape((bs,img_dim, img_dim,f))
        return img
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
        
    def create_patching_idxs(self,dim = 64, ps= 3*8, ps2 = 8):
        #ps = self.mult*ps2
        pad_dim = ps2 #int((ps-1)/2) 
        image_dim_pad = (2*pad_dim  + dim)
        arr = torch.tensor(list(range(image_dim_pad**2))).reshape((image_dim_pad,image_dim_pad)).long()
        idxs = torch.zeros((dim//ps2,dim//ps2,ps,ps)).long()
        for i in range(dim//ps2):
            for j in range(dim//ps2):
                    idxs[i,j,:,:] =  arr[i:i +ps ,j:j+ps]

        return idxs.flatten()
        
    def get_patches_mini(self,x):
        bs, h,w, c = x.shape
        padding = int((self.patch_size)**((self.mult-1)//2)) # padding = 3*8//2 =3*4 = 4
        y = F.pad(x.permute(0,3,1,2), (padding, padding, padding, padding), mode='constant', value=0)
        y = torch.index_select(y.flatten(-2), 2, self.patch_idxs.to(x.device))
        #print(y.shape)
        y =y.reshape(bs,c,self.num_patches ,self.patch_size_l,self.patch_size_l).permute(0,2,3,4,1)
        y = self.get_patches_flat(y.reshape(bs*self.num_patches ,self.patch_size_l,self.patch_size_l,c), ps=self.patch_size)
        y = y.reshape((bs, self.num_patches, self.mult*self.mult,self.patch_size*self.patch_size, c)).mean(2)
        #print(y.shape)
        return y.reshape((bs*self.num_patches,self.patch_size*self.patch_size, c)) #
 
        
    def forward(self, inp):
        
       
       
        if self.is_training:   
            with torch.no_grad():
                x , Vh = inp
               # print("x.shape",x.shape)
                x = x.permute(0,2,3,1)
                bs, idim, idim , cs = x.size() 
                x = x.reshape((bs, idim, idim,cs//3, 3)).permute(0,1,2,4,3)
          
                x = self.get_patches_flat(x[:,:,:,:,1:].flatten(-2),ps=self.patch_size)
                if len(Vh.shape) == 6:
                    Vh = Vh.squeeze(1)
        else:
             with torch.no_grad():
                x = inp.permute(0,2,3,1)
                bs, idim, idim , cs = x.size() 
                x = x.reshape((bs, idim, idim,cs//3, 3)).permute(0,1,2,4,3)
                x0 = self.get_patches_mini(x[:,:,:,:,:-1].flatten(-2))
                #print( x0.shape)
                x = self.get_patches_flat(x[:,:,:,:,1:].flatten(-2),ps=self.patch_size)
               # _,_,Vh,det_Vh= self.unique_svd(x0)
                _,_,Vh= self.unique_svd(x0)

              #  mask = det_Vh != 0
                # Expand the mask and determinant to have the same shape as Vh for broadcasting
              #  expanded_mask = mask[:, None, None].expand_as(Vh)
             #   expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
             
              #  Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
        

                Vh=Vh.reshape(bs,self.num_patches,1,6,6)
            
             
        y =  (x.reshape(bs,self.num_patches,self.patch_size**2,3,2)@self.weightTime[0]).permute(0,1,2,4,3).flatten(-2)
      #  print("y", y.shape)  
      #  print("Vh", Vh.shape)
        y =  torch.selu( torch.abs(y.unsqueeze(-2)@Vh.unsqueeze(0)).squeeze(-2) )# the abs needed as VhS defined up to signs 
               # print("y2.shape", y2.shape)
        try:
            Vh = Vh.unsqueeze(-3).repeat(1,1,1,self.embed_dim2,1,1 )
        except: 
            print("**** error Vh *** ", Vh.shape)
            print(" y.shape", y.shape)
            print(" x.shape", x.shape)
        
       # print("Vh", Vh.shape)
        x = x.reshape(bs,self.num_patches,self.patch_size**2,3,2).unsqueeze(-2).repeat(1,1,1,1,self.embed_dim2,1).unsqueeze(-2)
      #  print("x", x.shape)  
        weightsT = self.weightTime[1:].reshape(1,1,1,1,self.embed_dim2,2,2)
        #print("weightsT", weightsT.shape)  
        x =  (x@weightsT).permute(0,1,2,6,4,5,3) #
       # print("x", x.shape)  
      #  weightSym  =  torch.softmax(self.weightSym ,dim=-1)    
      #  weightSym  =  1/2*(weightSym + weightSym.permute(0,2,1)).reshape(1,1,1,1,self.embed_dim2//2,3,3)#.unsqueeze(-1).repeat(1,1,1,4)
        weightOrth =  self.euler_to_so3(self.weightOrth,self.embed_dim2).reshape(1,1,1,1,self.embed_dim2,3,3)
        #x2 = self.proj_input2(torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2)
        #x = self.proj_input3(torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2)
      #  print("weightSym",weightSym.shape)
       # x2 = torch.abs(x[:,:,:,:,:self.embed_dim2//2]@weightSym).permute(0,1,2,4,3,5,6).flatten(-3).unsqueeze(-2)
        
      #  print("Vh", Vh.shape)
       # print("x2", x2.shape) 
      #  x2 = torch.selu(x2@Vh).squeeze(-2).flatten(-2)
        
      #  x = torch.abs(x[:,:,:,:,self.embed_dim2//2:]@weightOrth).permute(0,1,2,4,3,5,6).flatten(-3).unsqueeze(-2)
        x = (x@weightOrth).permute(0,1,2,4,3,5,6).flatten(-3).unsqueeze(-2)
      #  print("x", x.shape) 
        x = torch.selu(torch.abs(x@Vh)).squeeze(-2).flatten(-2)
     #   print("x", x.shape) 
      #  print("x2", x2.shape)
       # print("y", y.shape)
       # x = torch.cat([torch.selu(self.proj_input2(x)),torch.selu(self.proj_input3(x2)),torch.selu(self.proj_input(y.squeeze(0)))], dim=-1).reshape(bs,self.num_patches , self.patch_size**2, self.embed_dim)
        x = torch.cat([x,y.squeeze(0)], dim=-1).reshape(bs,self.num_patches , self.patch_size**2, self.embed_dim)
       # x = self.reconstruct_image(nn.ReLU()(self.proj_inputf(x)), ps=self.patch_size, img_dim = idim).permute(0,3,1,2)
        x = self.reconstruct_image(x, ps=self.patch_size, img_dim = idim).permute(0,3,1,2)
       # print("x.shape",x.shape)
        
        x = self.layers(x)
      #  print("x.shape",x.shape)
        
        return x


class HeadCNN(nn.Module):
	def __init__(self, in_shape, num_layers=0, num_filters=32):
		super().__init__()
		self.layers = []
		for _ in range(0, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers.append(Flatten())
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(in_shape, self.layers)
		self.apply(weight_init)

	def forward(self, x):
		#print("x.shape",x.shape)
		return self.layers(x)


class Encoder(nn.Module):
	def __init__(self, shared, head, projection):
		super().__init__()
		self.shared = shared
		self.head = head
		self.projection = projection
		self.out_dim = projection.out_dim

	def forward(self, x, detach=False):
		x = self.shared(x)
		x = self.head(x)
		if detach:
			x = x.detach()
		return self.projection(x)


class Actor(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max):
		super().__init__()
		self.encoder = encoder
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.mlp = nn.Sequential(
			nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 2 * action_shape[0])
		)
		self.mlp.apply(weight_init)

	def forward(self, x, compute_pi=True, compute_log_pi=True, detach=False):
		x = self.encoder(x, detach)
		mu, log_std = self.mlp(x).chunk(2, dim=-1)
		log_std = torch.tanh(log_std)
		log_std = self.log_std_min + 0.5 * (
			self.log_std_max - self.log_std_min
		) * (log_std + 1)

		if compute_pi:
			std = log_std.exp()
			noise = torch.randn_like(mu)
			pi = mu + noise * std
		else:
			pi = None
			entropy = None

		if compute_log_pi:
			log_pi = gaussian_logprob(noise, log_std)
		else:
			log_pi = None

		mu, pi, log_pi = squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std


class QFunction(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim):
		super().__init__()
		self.trunk = nn.Sequential(
			nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		self.apply(weight_init)

	def forward(self, obs, action):
		assert obs.size(0) == action.size(0)
		return self.trunk(torch.cat([obs, action], dim=1))


class Critic(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.Q1 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)
		self.Q2 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)

	def forward(self, x, action, detach=False):
		x = self.encoder(x, detach)
		return self.Q1(x, action), self.Q2(x, action)


class CURLHead(nn.Module):
	def __init__(self, encoder):
		super().__init__()
		self.encoder = encoder
		self.W = nn.Parameter(torch.rand(encoder.out_dim, encoder.out_dim))

	def compute_logits(self, z_a, z_pos):
		"""
		Uses logits trick for CURL:
		- compute (B,B) matrix z_a (W z_pos.T)
		- positives are all diagonal elements
		- negatives are all other elements
		- to compute loss use multiclass cross entropy with identity matrix for labels
		"""
		Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
		logits = torch.matmul(z_a, Wz)  # (B,B)
		logits = logits - torch.max(logits, 1)[0][:, None]
		return logits


class InverseDynamics(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = nn.Sequential(
			nn.Linear(2*encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, action_shape[0])
		)
		self.apply(weight_init)

	def forward(self, x, x_next):
		h = self.encoder(x)
		h_next = self.encoder(x_next)
		joint_h = torch.cat([h, h_next], dim=1)
		return self.mlp(joint_h)


class SODAPredictor(nn.Module):
	def __init__(self, encoder, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = SODAMLP(
			encoder.out_dim, hidden_dim, encoder.out_dim
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(self.encoder(x))
