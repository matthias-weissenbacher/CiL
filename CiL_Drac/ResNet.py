import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ucb_rl2_meta.distributions import Categorical
from ucb_rl2_meta.utils import init

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))

init_tanh_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

class NNBase(nn.Module):
    """
    Actor-Critic network (base class)
    """
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size
    
    @property
    def output_size_eq(self):
        return self._hidden_size*2

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )




class MLPBase(NNBase):
    """
    Multi-Layer Perceptron
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        self.actor = nn.Sequential(
            init_tanh_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_tanh_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_tanh_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_tanh_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    """
    Flatten a tensor
    """
    def forward(self, x):
        return x.reshape(x.size(0), -1)
    
class BasicBlock(nn.Module):
    """
    Residual Network Block
    """
    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


    
class ResNetBase2l(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBase2l, self).__init__(recurrent, num_inputs, hidden_size)

        self.layer1 = self._make_layer(4*num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(8*num_inputs, 4*num_inputs)
        #self.proj_input2 = nn.Linear(32, num_inputs)
        

        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    
    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
            Vhinv = torch.linalg.pinv(Vh)
            det_Vh = torch.det(Vh)

        
        return U,Sv.to(device),Vh.to(device),Vhinv.to(device),det_Vh.to(device)
    
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
    

        if idim ==1:
            #dummy
            x=x
        else:
            
            x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
            x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
            #print( x0.shape)
            #x = self.get_patches_flat(x,ps=self.patch_size)
            _,Sv,Vh, Vhinv ,det_Vh = self.unique_svd(x0)
    
            mask = det_Vh != 0
            # Expand the mask and determinant to have the same shape as Vh for broadcasting
            expanded_mask = mask[:, None, None].expand_as(Vh)
            expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
            
            VhSinv  = torch.diag_embed(Sv)@Vhinv
            mask = Sv <= 1e-5   
            Sv[mask] = 10e9    
            VhS = (Vh@torch.diag_embed(1/Sv))
            
            Vh=Vh.unsqueeze(1).unsqueeze(1)
            Vhinv=Vhinv.unsqueeze(1).unsqueeze(1)
            VhS=VhS.unsqueeze(1).unsqueeze(1)
            VhSinv=VhSinv.unsqueeze(1).unsqueeze(1)
        
            y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
            y1 = torch.abs(x.unsqueeze(-2)@(Vhinv.transpose(-1,-2))).squeeze(-2)
            ya = torch.abs(x.unsqueeze(-2)@VhS).squeeze(-2) # the abs needed as VhS defined up to signs 
            y1a = torch.abs(x.unsqueeze(-2)@(VhSinv.transpose(-1,-2))).squeeze(-2)
            Vh=Vh.squeeze(1).squeeze(1)
            Vhinv=Vhinv.squeeze(1).squeeze(1)
            VhS=VhS.squeeze(1).squeeze(1)
            VhSinv=VhSinv.squeeze(1).squeeze(1)
            # Divide the matrices by their determinant only where the determinant is non-zero
            Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
            Vhinv[expanded_mask] = Vhinv[expanded_mask] * expanded_det_Vh[expanded_mask]
            VhS[expanded_mask] = VhS[expanded_mask] /  expanded_det_Vh[expanded_mask]
            VhSinv[expanded_mask] = VhSinv[expanded_mask] * expanded_det_Vh[expanded_mask]
            Vh=Vh.unsqueeze(1).unsqueeze(1)
            Vhinv=Vhinv.unsqueeze(1).unsqueeze(1)
            VhS=VhS.unsqueeze(1).unsqueeze(1)
            VhSinv=VhSinv.unsqueeze(1).unsqueeze(1)
            y2 = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
            y3 = torch.abs(x.unsqueeze(-2)@(Vhinv.transpose(-1,-2))).squeeze(-2)
            y2a = torch.abs(x.unsqueeze(-2)@VhS).squeeze(-2) # the abs needed as VhS defined up to signs 
            y3a = torch.abs(x.unsqueeze(-2)@(VhSinv.transpose(-1,-2))).squeeze(-2)
    
            #x = torch.cat([self.norm1(y),self.norm2(y1),y2,y3],dim=-1)
            x = torch.cat([torch.selu(y),torch.selu(y1),torch.selu(y2),torch.selu(y3),torch.selu(ya),torch.selu(y1a),torch.selu(y2a),torch.selu(y3a)],dim=-1)
            print("x.shape", x.shape)
            x = torch.selu(self.proj_input(x).reshape(bs, idim, idim , 4*cs)) 
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
                

import math  

class ResNetBase2lga(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBase2lga, self).__init__(recurrent, num_inputs, hidden_size)
        self.embed_dim = 32
        self.layer1 = self._make_layer(6*num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(2*num_inputs, 2*num_inputs)
        self.proj_inputS = nn.Linear((self.embed_dim//2)*num_inputs, 4*num_inputs)
        #self.proj_input2 = nn.Linear(32, num_inputs)
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        colors = 3
        
        self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim//4, colors,colors))
        nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightASym = nn.Parameter(torch.Tensor(self.embed_dim//4, colors,colors))
        nn.init.kaiming_uniform_(self.weightASym, a=math.sqrt(5))

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    
    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
            Vhinv = torch.linalg.pinv(Vh)
            det_Vh = torch.det(Vh)

        
        return U,Sv.to(device),Vh.to(device),Vhinv.to(device),det_Vh.to(device)
    
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
    

        if idim ==1:
            #dummy
            x=x
        else:
            
            x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
            x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
            #print( x0.shape)
            #x = self.get_patches_flat(x,ps=self.patch_size)
            _,Sv,Vh, Vhinv ,det_Vh = self.unique_svd(x0)
    
            mask = det_Vh != 0
            # Expand the mask and determinant to have the same shape as Vh for broadcasting
            expanded_mask = mask[:, None, None].expand_as(Vh)
            expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
            
           # VhSinv  = torch.diag_embed(Sv)@Vhinv
           # mask = Sv <= 1e-5   
          #  Sv[mask] = 10e9    
           # VhS = (Vh@torch.diag_embed(1/Sv))
      
            # Divide the matrices by their determinant only where the determinant is non-zero
            Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
            Vhinv[expanded_mask] = Vhinv[expanded_mask] * expanded_det_Vh[expanded_mask]
           # VhS[expanded_mask] = VhS[expanded_mask] /  expanded_det_Vh[expanded_mask]
            #VhSinv[expanded_mask] = VhSinv[expanded_mask] * expanded_det_Vh[expanded_mask]
            Vh=Vh.unsqueeze(1).unsqueeze(1)
            Vhinv2 = Vhinv.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,1,self.embed_dim//4,1,1 )
            Vhinv=Vhinv.unsqueeze(1).unsqueeze(1)
          #  VhS=VhS.unsqueeze(1).unsqueeze(1)
          #  VhSinv=VhSinv.unsqueeze(1).unsqueeze(1)
            y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
            y1 = torch.abs(x.unsqueeze(-2)@(Vhinv.transpose(-1,-2))).squeeze(-2)
      
            x2 = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim//4,1,1)
        

        weightASym  =  self.weightASym - self.weightASym.permute(0,2,1)  
        weightSym  =  self.weightSym + self.weightSym.permute(0,2,1)   
        y2 = torch.abs(x2@(weightSym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        y3 = torch.abs(x2@(weightASym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        z = torch.selu(self.proj_inputS(torch.cat([torch.selu(y2),torch.selu(y3)],dim=-1)))
        
        x =  torch.cat([torch.selu(y),torch.selu(y1)],dim=-1)
            #print("x.shape", x.shape)
        x = torch.cat([torch.selu(self.proj_input(x)),z],dim=-1).reshape(bs, idim, idim , 6*cs)
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
    
    
class ResNetBaseLocal2(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBaseLocal2, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.embed_dim2 = 20
        self.embed_dim = 3*self.embed_dim2 + 4
        self.layer1 = self._make_layer(channels[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(num_inputs, 4)
       # self.proj_input2 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
       # self.proj_input3 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
        #self.proj_inputS = nn.Linear(3*(self.embed_dim//4)*num_inputs, 6*num_inputs)
        self.projf = nn.Linear(self.embed_dim, channels[0])
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        in_chans = 3
        
                
        self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim2//2, in_chans,in_chans)) 
        nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightOrth = nn.Parameter(torch.Tensor(self.embed_dim2//2, 3))
        nn.init.kaiming_uniform_(self.weightOrth, a=math.sqrt(5))
        
        
        self.patch_size = 8
        self.mult = 5
        self.patch_size_l = self.mult*self.patch_size 
        self.idim = 64
        self.num_patches =  (self.idim //self.patch_size)**2
        self.patch_idxs = self.create_patching_idxs(dim = self.idim , ps= self.patch_size_l  , ps2 = self.patch_size ) 
        
        
        


    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    

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
            det_Vh = torch.det(Vh)

        
        return U,Sv,Vh.to(device),det_Vh.to(device)
         
         
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

        y = y.reshape((bs, self.num_patches, self.mult**2,self.patch_size*self.patch_size, c)).mean(2)
        #print(y.shape)
        return y.reshape((bs*self.num_patches,self.patch_size*self.patch_size, c)) #
 
    def reconstruct_image(self,patches, ps=8, img_dim = 64):
        xy_dim = img_dim // ps
        
        bs,num_p, num_pn, f = patches.size()    
        patches = patches.reshape(bs, num_p ,ps,ps,f)
        img = patches.reshape((bs,xy_dim,xy_dim,ps, ps,f)).permute(0,1,3,2,4,5)
        img = img.reshape((bs,img_dim, img_dim,f))
        return img
         
    
         
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
      
          
        with torch.no_grad():
                x0 = self.get_patches_mini(x)
                #print( x0.shape)
          
                x = self.get_patches_flat(x,ps=self.patch_size)
                _,_,Vh,det_Vh= self.unique_svd(x0)

                mask = det_Vh != 0
                # Expand the mask and determinant to have the same shape as Vh for broadcasting
                expanded_mask = mask[:, None, None].expand_as(Vh)
                expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
             
                Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
        

                Vh=Vh.unsqueeze(1)
             
             
     
                y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
               # print("y2.shape", y2.shape)
            
                Vh = Vh.unsqueeze(1).repeat(1,1,1,self.embed_dim2 //2,1,1 )
                x = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim2 //2 ,1,1)
        weightSym  =  torch.softmax(self.weightSym ,dim=-1)    
        weightSym  =  1/2*(weightSym + weightSym.permute(0,2,1))
        weightOrth =  self.euler_to_so3(self.weightOrth,self.embed_dim2//2)
       # x2 = nn.ReLU()(self.proj_input2(torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
      #  x = nn.ReLU()(self.proj_input3(torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
        x2 = torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2)).flatten(-2)
        x = torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2)).flatten(-2)
  
        x = nn.ReLU()(self.projf(torch.cat([x,x2,torch.selu(self.proj_input(y)).unsqueeze(0)], dim=-1).reshape(bs,self.num_patches , self.patch_size**2, self.embed_dim)))
            #x = x.reshape(bs, idim, idim , 4*cs) 

            
        x = self.reconstruct_image(x, ps=self.patch_size, img_dim = self.idim)
        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
    
    
class ResNetBaseLocal3(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBaseLocal3, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.embed_dim2 = 32
        self.embed_dim = 3*self.embed_dim2//2 + 8
        self.layer1 = self._make_layer(channels[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(num_inputs, 8)
       # self.proj_input2 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
       # self.proj_input3 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
        #self.proj_inputS = nn.Linear(3*(self.embed_dim//4)*num_inputs, 6*num_inputs)
        self.projf = nn.Linear(self.embed_dim, channels[0])
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        in_chans = 3
        
                
       # self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim2//2, in_chans,in_chans)) 
       # nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightOrth = nn.Parameter(torch.Tensor(self.embed_dim2//2, 3))
        nn.init.kaiming_uniform_(self.weightOrth, a=math.sqrt(5))
        
        
        self.patch_size = 8
        self.mult = 5
        self.patch_size_l = self.mult*self.patch_size 
        self.idim = 64
        self.num_patches =  (self.idim //self.patch_size)**2
        self.patch_idxs = self.create_patching_idxs(dim = self.idim , ps= self.patch_size_l  , ps2 = self.patch_size ) 
        
        
        


    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    

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
            det_Vh = torch.det(Vh)

        
        return U,Sv,Vh.to(device),det_Vh.to(device)
         
         
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

        y = y.reshape((bs, self.num_patches, self.mult**2,self.patch_size*self.patch_size, c)).mean(2)
        #print(y.shape)
        return y.reshape((bs*self.num_patches,self.patch_size*self.patch_size, c)) #
 
    def reconstruct_image(self,patches, ps=8, img_dim = 64):
        xy_dim = img_dim // ps
        
        bs,num_p, num_pn, f = patches.size()    
        patches = patches.reshape(bs, num_p ,ps,ps,f)
        img = patches.reshape((bs,xy_dim,xy_dim,ps, ps,f)).permute(0,1,3,2,4,5)
        img = img.reshape((bs,img_dim, img_dim,f))
        return img
         
    
         
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
      
          
        with torch.no_grad():
                x0 = self.get_patches_mini(x)
                #print( x0.shape)
          
                x = self.get_patches_flat(x,ps=self.patch_size)
                _,_,Vh,det_Vh= self.unique_svd(x0)

                mask = det_Vh != 0
                # Expand the mask and determinant to have the same shape as Vh for broadcasting
                expanded_mask = mask[:, None, None].expand_as(Vh)
                expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
             
              #  Vh2 = Vh.clone().unsqueeze(1)
                Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
        

                Vh=Vh.unsqueeze(1)
             
             
     
                y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
               # print("y2.shape", y2.shape)
            
                Vh = Vh.unsqueeze(1).repeat(1,1,1,self.embed_dim2 //2,1,1 )
              #  Vh2 = Vh2.unsqueeze(1).repeat(1,1,1,self.embed_dim2 //2,1,1 )
                x = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim2 //2 ,1,1)
     #   weightSym  =  torch.softmax(self.weightSym ,dim=-1)    
       # weightSym  =  1/2*(weightSym + weightSym.permute(0,2,1))
        weightOrth =  self.euler_to_so3(self.weightOrth,self.embed_dim2//2)
       # x2 = nn.ReLU()(self.proj_input2(torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
      #  x = nn.ReLU()(self.proj_input3(torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
       # x2 = torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2)).flatten(-2)
        x = torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2)).flatten(-2)
  
        x = nn.ReLU()(self.projf(torch.cat([x,torch.selu(self.proj_input(y)).unsqueeze(0)], dim=-1).reshape(bs,self.num_patches , self.patch_size**2, self.embed_dim)))
            #x = x.reshape(bs, idim, idim , 4*cs) 

            
        x = self.reconstruct_image(x, ps=self.patch_size, img_dim = self.idim)
        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
        
    
class ResNetBaseLocal4(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBaseLocal4, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.embed_dim2 = 32
        self.embed_dim = 3*self.embed_dim2//2 + 8
        self.layer1 = self._make_layer(channels[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(num_inputs, 8)
       # self.proj_input2 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
       # self.proj_input3 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
        #self.proj_inputS = nn.Linear(3*(self.embed_dim//4)*num_inputs, 6*num_inputs)
        self.projf = nn.Linear(self.embed_dim, channels[0])
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        in_chans = 3
        
                
       # self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim2//2, in_chans,in_chans)) 
       # nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightOrth = nn.Parameter(torch.Tensor(self.embed_dim2//2, 3))
        nn.init.kaiming_uniform_(self.weightOrth, a=math.sqrt(5))
        
        
        self.patch_size = 8
        self.mult = 5
        self.patch_size_l = self.mult*self.patch_size 
        self.idim = 64
        self.num_patches =  (self.idim //self.patch_size)**2
        self.patch_idxs = self.create_patching_idxs(dim = self.idim , ps= self.patch_size_l  , ps2 = self.patch_size ) 
        
        
        


    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    

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

        
        return U,Sv,Vh.to(device) #,det_Vh.to(device)
         
         
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

        y = y.reshape((bs, self.num_patches, self.mult**2,self.patch_size*self.patch_size, c)).mean(2)
        #print(y.shape)
        return y.reshape((bs*self.num_patches,self.patch_size*self.patch_size, c)) #
 
    def reconstruct_image(self,patches, ps=8, img_dim = 64):
        xy_dim = img_dim // ps
        
        bs,num_p, num_pn, f = patches.size()    
        patches = patches.reshape(bs, num_p ,ps,ps,f)
        img = patches.reshape((bs,xy_dim,xy_dim,ps, ps,f)).permute(0,1,3,2,4,5)
        img = img.reshape((bs,img_dim, img_dim,f))
        return img
         
    
         
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
      
          
        with torch.no_grad():
                x0 = self.get_patches_mini(x)
                #print( x0.shape)
          
                x = self.get_patches_flat(x,ps=self.patch_size)
                _,_,Vh = self.unique_svd(x0)

              #  mask = det_Vh != 0
                # Expand the mask and determinant to have the same shape as Vh for broadcasting
              #  expanded_mask = mask[:, None, None].expand_as(Vh)
               # expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
             
              #  Vh2 = Vh.clone().unsqueeze(1)
              #  Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
        

                Vh=Vh.unsqueeze(1)
             
             
     
                y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
               # print("y2.shape", y2.shape)
            
                Vh = Vh.unsqueeze(1).repeat(1,1,1,self.embed_dim2 //2,1,1 )
              #  Vh2 = Vh2.unsqueeze(1).repeat(1,1,1,self.embed_dim2 //2,1,1 )
                x = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim2 //2 ,1,1)
     #   weightSym  =  torch.softmax(self.weightSym ,dim=-1)    
       # weightSym  =  1/2*(weightSym + weightSym.permute(0,2,1))
        weightOrth =  self.euler_to_so3(self.weightOrth,self.embed_dim2//2)
       # x2 = nn.ReLU()(self.proj_input2(torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
      #  x = nn.ReLU()(self.proj_input3(torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
       # x2 = torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2)).flatten(-2)
        x = torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2)).flatten(-2)
  
        x = nn.ReLU()(self.projf(torch.cat([x,torch.selu(self.proj_input(y)).unsqueeze(0)], dim=-1).reshape(bs,self.num_patches , self.patch_size**2, self.embed_dim)))
            #x = x.reshape(bs, idim, idim , 4*cs) 

            
        x = self.reconstruct_image(x, ps=self.patch_size, img_dim = self.idim)
        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
        
        
        
class ResNetBaseGlobal(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBaseGlobal, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.embed_dim2 = 20
        self.embed_dim = 3*self.embed_dim2 + 4
        self.layer1 = self._make_layer(channels[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(num_inputs, 4)
       # self.proj_input2 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
       # self.proj_input3 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
        #self.proj_inputS = nn.Linear(3*(self.embed_dim//4)*num_inputs, 6*num_inputs)
        self.projf = nn.Linear(self.embed_dim, channels[0])
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        in_chans = 3
        
                
        self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim2//2, in_chans,in_chans)) 
        nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightOrth = nn.Parameter(torch.Tensor(self.embed_dim2//2, 3))
        nn.init.kaiming_uniform_(self.weightOrth, a=math.sqrt(5))
        


    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    

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
            det_Vh = torch.det(Vh)

        
        return U,Sv,Vh.to(device),det_Vh.to(device)
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
          
        with torch.no_grad():
               # x0 = self.get_patches_mini(x)
                
                x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
                x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
               # x = self.get_patches_flat(x,ps=self.patch_size)
                _,_,Vh,det_Vh= self.unique_svd(x0)

                mask = det_Vh != 0
                # Expand the mask and determinant to have the same shape as Vh for broadcasting
                expanded_mask = mask[:, None, None].expand_as(Vh)
                expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
             
                Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
        

                Vh=Vh.unsqueeze(1).unsqueeze(1)
             
     
                y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
               # print("y2.shape", y2.shape)
            
                Vh = Vh.unsqueeze(1).repeat(1,1,1,self.embed_dim2 //2,1,1 )
                x = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim2 //2 ,1,1)
        weightSym  =  torch.softmax(self.weightSym ,dim=-1)    
        weightSym  =  1/2*(weightSym + weightSym.permute(0,2,1))
        weightOrth =  self.euler_to_so3(self.weightOrth,self.embed_dim2//2)
       # x2 = nn.ReLU()(self.proj_input2(torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
      #  x = nn.ReLU()(self.proj_input3(torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
        x2 = torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2)).flatten(-2)
        x = torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2)).flatten(-2)
     
        x = nn.ReLU()(self.projf(torch.cat([x,x2,torch.selu(self.proj_input(y))], dim=-1).reshape(bs,idim, idim , self.embed_dim)))
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
    
    
    
    
class ResNetBaseGlobal2(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBaseGlobal2, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.embed_dim2 = 20
        self.embed_dim = 3*self.embed_dim2 + 4
        self.layer1 = self._make_layer(channels[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(num_inputs, 4)
       # self.proj_input2 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
       # self.proj_input3 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
        #self.proj_inputS = nn.Linear(3*(self.embed_dim//4)*num_inputs, 6*num_inputs)
        self.projf = nn.Linear(self.embed_dim, channels[0])
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        in_chans = 3
        
                
        self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim2, in_chans,in_chans)) 
        nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
       # self.weightOrth = nn.Parameter(torch.Tensor(self.embed_dim2//2, 3))
       # nn.init.kaiming_uniform_(self.weightOrth, a=math.sqrt(5))
        


    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    

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
            det_Vh = torch.det(Vh)

        
        return U,Sv,Vh.to(device),det_Vh.to(device)
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
          
        with torch.no_grad():
               # x0 = self.get_patches_mini(x)
                
                x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
                x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
               # x = self.get_patches_flat(x,ps=self.patch_size)
                _,_,Vh,det_Vh= self.unique_svd(x0)

                mask = det_Vh != 0
                # Expand the mask and determinant to have the same shape as Vh for broadcasting
                expanded_mask = mask[:, None, None].expand_as(Vh)
                expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
             
                Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
        

                Vh=Vh.unsqueeze(1).unsqueeze(1)
             
     
                y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
               # print("y2.shape", y2.shape)
            
                Vh = Vh.unsqueeze(1).repeat(1,1,1,self.embed_dim2,1,1 )
                x = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim2,1,1)
        weightSym  =  torch.softmax(self.weightSym ,dim=-1)    
        weightSym  =  1/2*(weightSym + weightSym.permute(0,2,1))
       # weightOrth =  self.euler_to_so3(self.weightOrth,self.embed_dim2//2)
       # x2 = nn.ReLU()(self.proj_input2(torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
      #  x = nn.ReLU()(self.proj_input3(torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
        x = torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2)).flatten(-2)
        #x = torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2)).flatten(-2)
     
        x = nn.ReLU()(self.projf(torch.cat([x,torch.selu(self.proj_input(y))], dim=-1).reshape(bs,idim, idim , self.embed_dim)))
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
                
        
        
    
    
    
class ResNetBaseGlobal3(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBaseGlobal3, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.embed_dim2 = 20
        self.embed_dim = 3*self.embed_dim2 + 4
        self.layer1 = self._make_layer(channels[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(num_inputs, 4)
       # self.proj_input2 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
       # self.proj_input3 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
        #self.proj_inputS = nn.Linear(3*(self.embed_dim//4)*num_inputs, 6*num_inputs)
        self.projf = nn.Linear(self.embed_dim, channels[0])
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        in_chans = 3
        
                
        #self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim2, in_chans,in_chans)) 
        #nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightOrth = nn.Parameter(torch.Tensor(self.embed_dim2, 3))
        nn.init.kaiming_uniform_(self.weightOrth, a=math.sqrt(5))
        


    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    

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
            det_Vh = torch.det(Vh)

        
        return U,Sv,Vh.to(device),det_Vh.to(device)
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
          
        with torch.no_grad():
               # x0 = self.get_patches_mini(x)
                
                x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
                x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
               # x = self.get_patches_flat(x,ps=self.patch_size)
                _,_,Vh,det_Vh= self.unique_svd(x0)

                mask = det_Vh != 0
                # Expand the mask and determinant to have the same shape as Vh for broadcasting
                expanded_mask = mask[:, None, None].expand_as(Vh)
                expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
             
                Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
        

                Vh=Vh.unsqueeze(1).unsqueeze(1)
             
     
                y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
               # print("y2.shape", y2.shape)
            
                Vh = Vh.unsqueeze(1).repeat(1,1,1,self.embed_dim2 ,1,1 )
                x = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim2  ,1,1)
       # weightSym  =  torch.softmax(self.weightSym ,dim=-1)    
      #  weightSym  =  1/2*(weightSym + weightSym.permute(0,2,1))
        weightOrth =  self.euler_to_so3(self.weightOrth,self.embed_dim2//2)
       # x2 = nn.ReLU()(self.proj_input2(torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
      #  x = nn.ReLU()(self.proj_input3(torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
        #x = torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2)).flatten(-2)
        x = torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2)).flatten(-2)
     
        x = nn.ReLU()(self.projf(torch.cat([x,torch.selu(self.proj_input(y))], dim=-1).reshape(bs,idim, idim , self.embed_dim)))
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
    
    
    
class ResNetBaseGlobal4(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBaseGlobal4, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.embed_dim2 = 5
        self.embed_dim = 3*self.embed_dim2
        self.layer1 = self._make_layer( self.embed_dim , channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(num_inputs, 4)
       # self.proj_input2 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
       # self.proj_input3 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
        #self.proj_inputS = nn.Linear(3*(self.embed_dim//4)*num_inputs, 6*num_inputs)
        self.projf = nn.Linear(self.embed_dim, channels[0])
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        in_chans = 3
        
                
        #self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim2, in_chans,in_chans)) 
        #nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightOrth = nn.Parameter(torch.Tensor(self.embed_dim2, 3))
        nn.init.kaiming_uniform_(self.weightOrth, a=math.sqrt(5))
        


    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    

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
            det_Vh = torch.det(Vh)

        
        return U,Sv,Vh.to(device),det_Vh.to(device)
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
          
        with torch.no_grad():
               # x0 = self.get_patches_mini(x)
                
                x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
                x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
               # x = self.get_patches_flat(x,ps=self.patch_size)
                _,_,Vh,det_Vh= self.unique_svd(x0)

                mask = det_Vh != 0
                # Expand the mask and determinant to have the same shape as Vh for broadcasting
                expanded_mask = mask[:, None, None].expand_as(Vh)
                expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
             
                Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
        

                Vh=Vh.unsqueeze(1).unsqueeze(1)
             
     
                y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
               # print("y2.shape", y2.shape)
            
                Vh = Vh.unsqueeze(1).repeat(1,1,1,self.embed_dim2 ,1,1 )
                x = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim2  ,1,1)
       # weightSym  =  torch.softmax(self.weightSym ,dim=-1)    
      #  weightSym  =  1/2*(weightSym + weightSym.permute(0,2,1))
        weightOrth =  self.euler_to_so3(self.weightOrth,self.embed_dim2)
       # x2 = nn.ReLU()(self.proj_input2(torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
      #  x = nn.ReLU()(self.proj_input3(torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
        #x = torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2)).flatten(-2)
        x = torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2)).flatten(-2)
     
        x = x.reshape(bs,idim, idim , self.embed_dim)
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class ResNetBaseGlobal4a(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBaseGlobal4a, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.embed_dim2 = 5
        self.embed_dim = 3*self.embed_dim2
        self.layer1 = self._make_layer( self.embed_dim , channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(num_inputs, 4)
       # self.proj_input2 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
       # self.proj_input3 = nn.Linear(self.embed_dim2 //2,self.embed_dim2 //4)
        #self.proj_inputS = nn.Linear(3*(self.embed_dim//4)*num_inputs, 6*num_inputs)
        self.projf = nn.Linear(self.embed_dim, channels[0])
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        in_chans = 3
        
                
        #self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim2, in_chans,in_chans)) 
        #nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightOrth = nn.Parameter(torch.Tensor(self.embed_dim2, 3))
        nn.init.kaiming_uniform_(self.weightOrth, a=math.sqrt(5))
        


    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    

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

        
        return U,Sv,Vh.to(device) #,det_Vh.to(device)
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
          
        with torch.no_grad():
               # x0 = self.get_patches_mini(x)
                
                x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
                x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
               # x = self.get_patches_flat(x,ps=self.patch_size)
                _,_,Vh = self.unique_svd(x0)

               # mask = det_Vh != 0
                # Expand the mask and determinant to have the same shape as Vh for broadcasting
              #  expanded_mask = mask[:, None, None].expand_as(Vh)
              #  expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
             
             #   Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
        

                Vh=Vh.unsqueeze(1).unsqueeze(1)
             
     
                y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
               # print("y2.shape", y2.shape)
            
                Vh = Vh.unsqueeze(1).repeat(1,1,1,self.embed_dim2 ,1,1 )
                x = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim2  ,1,1)
       # weightSym  =  torch.softmax(self.weightSym ,dim=-1)    
      #  weightSym  =  1/2*(weightSym + weightSym.permute(0,2,1))
        weightOrth =  self.euler_to_so3(self.weightOrth,self.embed_dim2)
       # x2 = nn.ReLU()(self.proj_input2(torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
      #  x = nn.ReLU()(self.proj_input3(torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2).transpose(-1,-2))).flatten(-2))
        #x = torch.selu(torch.abs(x@(weightSym@Vh)).squeeze(-2)).flatten(-2)
        x = torch.selu(torch.abs(x@(weightOrth@Vh)).squeeze(-2)).flatten(-2)
     
        x = x.reshape(bs,idim, idim , self.embed_dim)
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
    
    
class ResNetBase2lt(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBase2lga, self).__init__(recurrent, num_inputs, hidden_size)
        self.embed_dim = 32
        self.layer1 = self._make_layer(6*num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(2*num_inputs, 2*num_inputs)
        self.proj_inputS = nn.Linear((self.embed_dim//2)*num_inputs, 4*num_inputs)
        #self.proj_input2 = nn.Linear(32, num_inputs)
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        colors = 3
        
     #   self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim//4, colors,colors))
     #   nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
     #   self.weightASym = nn.Parameter(torch.Tensor(self.embed_dim//4, colors,colors))
      #  nn.init.kaiming_uniform_(self.weightASym, a=math.sqrt(5))
        
        self.weightTU = nn.Parameter(torch.Tensor(self.embed_dim//4, colors,colors))
        nn.init.kaiming_uniform_(self.weightTU, a=math.sqrt(5))
        self.weightTL = nn.Parameter(torch.Tensor(self.embed_dim//4, colors,colors))
        nn.init.kaiming_uniform_(self.weightTL, a=math.sqrt(5))

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    
    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
            Vhinv = torch.linalg.pinv(Vh)
            det_Vh = torch.det(Vh)

        
        return U,Sv.to(device),Vh.to(device),Vhinv.to(device),det_Vh.to(device)
    
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
    

        if idim ==1:
            #dummy
            x=x
        else:
            
            x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
            x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
            #print( x0.shape)
            #x = self.get_patches_flat(x,ps=self.patch_size)
            _,Sv,Vh, Vhinv ,det_Vh = self.unique_svd(x0)
    
            mask = det_Vh != 0
            # Expand the mask and determinant to have the same shape as Vh for broadcasting
            expanded_mask = mask[:, None, None].expand_as(Vh)
            expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
            
           # VhSinv  = torch.diag_embed(Sv)@Vhinv
           # mask = Sv <= 1e-5   
          #  Sv[mask] = 10e9    
           # VhS = (Vh@torch.diag_embed(1/Sv))
      
            # Divide the matrices by their determinant only where the determinant is non-zero
            Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
            Vhinv[expanded_mask] = Vhinv[expanded_mask] * expanded_det_Vh[expanded_mask]
           # VhS[expanded_mask] = VhS[expanded_mask] /  expanded_det_Vh[expanded_mask]
            #VhSinv[expanded_mask] = VhSinv[expanded_mask] * expanded_det_Vh[expanded_mask]
            Vh=Vh.unsqueeze(1).unsqueeze(1)
            Vhinv2 = Vhinv.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,1,self.embed_dim//4,1,1 )
            Vhinv=Vhinv.unsqueeze(1).unsqueeze(1)
          #  VhS=VhS.unsqueeze(1).unsqueeze(1)
          #  VhSinv=VhSinv.unsqueeze(1).unsqueeze(1)
            y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
            y1 = torch.abs(x.unsqueeze(-2)@(Vhinv.transpose(-1,-2))).squeeze(-2)
      
            x2 = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim//4,1,1)
        

        weightTU =  torch.triu(self.weightTU)
        weightTL = torch.tril(self.weightTL)
        y2 = torch.abs(x2@(weightTU@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        y3 = torch.abs(x2@(weightTL@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        z = torch.selu(self.proj_inputS(torch.cat([torch.selu(y2),torch.selu(y3)],dim=-1)))
        
        x =  torch.cat([torch.selu(y),torch.selu(y1)],dim=-1)
            #print("x.shape", x.shape)
        x = torch.cat([torch.selu(self.proj_input(x)),z],dim=-1).reshape(bs, idim, idim , 6*cs)
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
    
#def func_a(a):
 #   return torch.sqrt(torch.abs(1+(2-3*a)*a))

def get_idxs(dim = 64) :
    #dims = [4,2,5,4] 
    num = 15
    
    idxls_base= torch.tensor([[0,1,2,1,0,2,2,2,3],[4,5,5,5,4,5,5,5,4],[6,7,7,8,9,10,8,10,9],[11,12,13,12,14,12,13,12,11]])
    idxls = torch.zeros((4*(dim//4),9))
    idxls[0:4] =  idxls_base
    for i in range(1,dim//4):
             idxls[4*i:(i+1)*4] = idxls_base + i*num 
            
    return idxls.long()

import math

class ResNetBasePISym(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):# channels=[32,32,32]
        super(ResNetBasePISym, self).__init__(recurrent, num_inputs, hidden_size)
        self.embed_dim = 16
        self.layer1 = self._make_layer(self.embed_dim, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(num_inputs, 4)
        self.proj_inputS = nn.Linear((self.embed_dim)*num_inputs, self.embed_dim-4)
        #self.proj_input2 = nn.Linear(32, num_inputs)
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        self.idxs = get_idxs(dim = self.embed_dim)

        

        self.weights = torch.nn.Parameter(torch.Tensor((self.embed_dim //4),15))  # self.embed_dim // 4 as 4 PI 
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init



        apply_init_(self.modules())

        self.train()
        
        colors = 3
        


    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    
    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
           
        
        return U,Sv.to(device),Vh.to(device)
    
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
    

        if idim ==1:
            #dummy
            x=x
        else:
            
            x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
            x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
            #print( x0.shape)
            #x = self.get_patches_flat(x,ps=self.patch_size)
            _,Sv,Vh,  = self.unique_svd(x0)
    
          
            Vh= Vh.unsqueeze(1).unsqueeze(1)
            
            y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
            Vh = Vh.unsqueeze(1).repeat(1,1,1,self.embed_dim,1,1 )
            x2 = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim,1,1)
        
        W = self.weights.flatten()[self.idxs.flatten()].reshape(self.embed_dim ,3,3)
        
        y2 = torch.abs(x2@(W@Vh)).squeeze(-2).flatten(-2)
       
        x = torch.cat([torch.selu(self.proj_input(torch.selu(y))),torch.selu(self.proj_inputS(torch.selu(y2)))],dim=-1).reshape(bs, idim, idim , self.embed_dim)
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs    
    
    
class ResNetBasePISym2(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBasePISym2, self).__init__(recurrent, num_inputs, hidden_size)
        self.embed_dim = 16
        self.layer1 = self._make_layer(self.embed_dim, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(num_inputs, 4)
        self.proj_inputS = nn.Linear((self.embed_dim)*num_inputs, self.embed_dim//2-2)
        self.proj_inputS2 = nn.Linear((self.embed_dim)*num_inputs, self.embed_dim//2-2)
        #self.proj_input2 = nn.Linear(32, num_inputs)
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        self.idxs = get_idxs(dim = self.embed_dim)

        

        self.weights = torch.nn.Parameter(torch.Tensor((self.embed_dim //4),15))  # self.embed_dim // 4 as 4 PI 
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init

        self.weights2 =  torch.nn.Parameter(torch.Tensor((self.embed_dim),3))
        torch.nn.init.kaiming_uniform_(self.weights2, a=math.sqrt(5)) 

        apply_init_(self.modules())

        self.train()
        
        colors = 3
        


    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    
    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
           
        
        return U,Sv.to(device),Vh.to(device)
    
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
    

        if idim ==1:
            #dummy
            x=x
        else:
            
            x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
            x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
            #print( x0.shape)
            #x = self.get_patches_flat(x,ps=self.patch_size)
            _,Sv,Vh,  = self.unique_svd(x0)
    
          
            Vh= Vh.unsqueeze(1).unsqueeze(1)
            
            y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
            Vh = Vh.unsqueeze(1).repeat(1,1,1,self.embed_dim,1,1 )
            x2 = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim,1,1)
        
        W = self.weights.flatten()[self.idxs.flatten()].reshape(self.embed_dim ,3,3)
        W2 =torch.diag_embed(self.weights2).reshape(self.embed_dim ,3,3)
        
        y2 = self.proj_inputS(torch.abs(x2@(W@Vh)).squeeze(-2).flatten(-2))
        y3 = self.proj_inputS2(torch.abs(x2@(W2@Vh)).squeeze(-2).flatten(-2))
        x = torch.cat([torch.selu(y2), torch.selu(y3)],dim=-1)
                                           
        x = torch.cat([torch.selu(self.proj_input(torch.selu(y))),torch.selu(x)],dim=-1).reshape(bs, idim, idim , self.embed_dim)
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs    
    
        
    
class ResNetBaseTriSym(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[32,32,32]):#resnet tiny
        super(ResNetBaseTriSym, self).__init__(recurrent, num_inputs, hidden_size)
        self.embed_dim = 32
        self.layer1 = self._make_layer(channels[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(2*num_inputs, 8)
        self.proj_inputS = nn.Linear((self.embed_dim)*num_inputs, 24)
        #self.proj_input2 = nn.Linear(32, num_inputs)
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        colors = 3
        
        self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim//4, colors,colors))
        nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightASym = nn.Parameter(torch.Tensor(self.embed_dim//4, colors,colors))
        nn.init.kaiming_uniform_(self.weightASym, a=math.sqrt(5))
        
        self.weightTU = nn.Parameter(torch.Tensor(self.embed_dim//4 ,colors,colors))
        nn.init.kaiming_uniform_(self.weightTU, a=math.sqrt(5))
        self.weightTL = nn.Parameter(torch.Tensor(self.embed_dim//4, colors,colors))
        nn.init.kaiming_uniform_(self.weightTL, a=math.sqrt(5))

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    
    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
            Vhinv = torch.linalg.pinv(Vh)
            det_Vh = torch.det(Vh)

        
        return U,Sv.to(device),Vh.to(device),Vhinv.to(device),det_Vh.to(device)
    
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
    

        if idim ==1:
            #dummy
            x=x
        else:
            
            x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
            x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
            #print( x0.shape)
            #x = self.get_patches_flat(x,ps=self.patch_size)
            _,Sv,Vh, Vhinv ,det_Vh = self.unique_svd(x0)
    
            mask = det_Vh != 0
            # Expand the mask and determinant to have the same shape as Vh for broadcasting
            expanded_mask = mask[:, None, None].expand_as(Vh)
            expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
            
           # VhSinv  = torch.diag_embed(Sv)@Vhinv
           # mask = Sv <= 1e-5   
          #  Sv[mask] = 10e9    
           # VhS = (Vh@torch.diag_embed(1/Sv))
      
            # Divide the matrices by their determinant only where the determinant is non-zero
            Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
            Vhinv[expanded_mask] = Vhinv[expanded_mask] * expanded_det_Vh[expanded_mask]
           # VhS[expanded_mask] = VhS[expanded_mask] /  expanded_det_Vh[expanded_mask]
            #VhSinv[expanded_mask] = VhSinv[expanded_mask] * expanded_det_Vh[expanded_mask]
            Vh=Vh.unsqueeze(1).unsqueeze(1)
            Vhinv2 = Vhinv.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,1,self.embed_dim//4,1,1 )
            Vhinv=Vhinv.unsqueeze(1).unsqueeze(1)
          #  VhS=VhS.unsqueeze(1).unsqueeze(1)
          #  VhSinv=VhSinv.unsqueeze(1).unsqueeze(1)
            y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
            y1 = torch.abs(x.unsqueeze(-2)@(Vhinv.transpose(-1,-2))).squeeze(-2)
      
            x2 = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim//4,1,1)
        

        weightTU =  torch.triu(self.weightTU)
        weightTL = torch.tril(self.weightTL)
        weightASym  =  self.weightASym - self.weightASym.permute(0,2,1)  
        weightSym  =  self.weightSym + self.weightSym.permute(0,2,1)   
        y2 = torch.abs(x2@(weightSym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        y3 = torch.abs(x2@(weightASym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        y4 = torch.abs(x2@(weightTU@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        y5 = torch.abs(x2@(weightTL@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        z = torch.selu(self.proj_inputS(torch.cat([torch.selu(y2),torch.selu(y3),torch.selu(y4),torch.selu(y5)],dim=-1)))
        
        x =  torch.cat([torch.selu(y),torch.selu(y1)],dim=-1)
            #print("x.shape", x.shape)
        x = torch.cat([torch.selu(self.proj_input(x)),z],dim=-1).reshape(bs, idim, idim , 32)
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs    
    
    
    
    
    
    
    
    
class ResNetBase2lt0(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[64,32,32]):#resnet tiny
        super(ResNetBase2lt0, self).__init__(recurrent, num_inputs, hidden_size)
        self.embed_dim = 10
        self.layer1 = self._make_layer(channels[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(2*num_inputs, 4)
        #self.proj_inputS = nn.Linear((self.embed_dim//2)*num_inputs, 4*num_inputs)
        #self.proj_input2 = nn.Linear(32, num_inputs)
        
   
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        colors = 3
        
        self.weightTU = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightTU, a=math.sqrt(5))
        self.weightTL = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightTL, a=math.sqrt(5))

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    
    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
            Vhinv = torch.linalg.pinv(Vh)
            det_Vh = torch.det(Vh)

        
        return U,Sv.to(device),Vh.to(device),Vhinv.to(device),det_Vh.to(device)
    
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
    

        if idim ==1:
            #dummy
            x=x
        else:
            
            x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
            x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
            #print( x0.shape)
            #x = self.get_patches_flat(x,ps=self.patch_size)
            _,Sv,Vh, Vhinv ,det_Vh = self.unique_svd(x0)
    
            mask = det_Vh != 0
            # Expand the mask and determinant to have the same shape as Vh for broadcasting
            expanded_mask = mask[:, None, None].expand_as(Vh)
            expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
            
           # VhSinv  = torch.diag_embed(Sv)@Vhinv
           # mask = Sv <= 1e-5   
          #  Sv[mask] = 10e9    
           # VhS = (Vh@torch.diag_embed(1/Sv))
      
            # Divide the matrices by their determinant only where the determinant is non-zero
            Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
            Vhinv[expanded_mask] = Vhinv[expanded_mask] * expanded_det_Vh[expanded_mask]
           # VhS[expanded_mask] = VhS[expanded_mask] /  expanded_det_Vh[expanded_mask]
            #VhSinv[expanded_mask] = VhSinv[expanded_mask] * expanded_det_Vh[expanded_mask]
            Vh=Vh.unsqueeze(1).unsqueeze(1)
            Vhinv2 = Vhinv.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,1,self.embed_dim,1,1 )
            Vhinv=Vhinv.unsqueeze(1).unsqueeze(1)
          #  VhS=VhS.unsqueeze(1).unsqueeze(1)
          #  VhSinv=VhSinv.unsqueeze(1).unsqueeze(1)
            y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
            y1 = torch.abs(x.unsqueeze(-2)@(Vhinv.transpose(-1,-2))).squeeze(-2)
      
            x2 = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim,1,1)
        

        weightTU = torch.softmax( torch.triu(self.weightTU),dim=1)
        weightTL = torch.softmax( torch.tril(self.weightTL),dim=1) 
        y2 = torch.abs(x2@(weightTU@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        y3 = torch.abs(x2@(weightTL@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        #z = torch.selu(self.proj_inputS(torch.cat([torch.selu(y2),torch.selu(y3)],dim=-1)))
        z = torch.cat([torch.selu(y2),torch.selu(y3)],dim=-1)
        
        x =  torch.cat([torch.selu(y),torch.selu(y1)],dim=-1)
            #print("x.shape", x.shape)
        x = torch.cat([torch.selu(self.proj_input(x)),z],dim=-1).reshape(bs, idim, idim , 64)
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
                
                
        
class ResNetBase2lgat(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[64,32,32]):#resnet tiny
        super(ResNetBase2lgat, self).__init__(recurrent, num_inputs, hidden_size)
        self.embed_dim = 5 # channels[0] -8
        self.layer1 = self._make_layer(channels[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(2*num_inputs, 4)
     #   self.proj_inputS = nn.Linear(self.embed_dim*num_inputs, channels[0]-8)
    #    self.proj_inputS1 = nn.Linear(self.embed_dim*num_inputs, 7)
      #  self.proj_inputS2 = nn.Linear(self.embed_dim*num_inputs, 7)
       # self.proj_inputS3 = nn.Linear(self.embed_dim*num_inputs, 7)
     #   self.proj_inputS4 = nn.Linear(self.embed_dim*num_inputs, 7)
       
        
        self.finalPooling = nn.MaxPool2d((2,2), stride=(2,2))

        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048//4, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        colors = 3
        
        self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightASym = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightASym, a=math.sqrt(5))
        self.weightTU = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightTU, a=math.sqrt(5))
        self.weightTL = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightTL, a=math.sqrt(5))
        

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    
    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
            Vhinv = torch.linalg.pinv(Vh)
            det_Vh = torch.det(Vh)

        
        return U,Sv.to(device),Vh.to(device),Vhinv.to(device),det_Vh.to(device)
    
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
    

        if idim ==1:
            #dummy
            x=x
        else:
            
            x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
            x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
            #print( x0.shape)
            #x = self.get_patches_flat(x,ps=self.patch_size)
            _,Sv,Vh, Vhinv ,det_Vh = self.unique_svd(x0)
    
            mask = det_Vh != 0
            # Expand the mask and determinant to have the same shape as Vh for broadcasting
            expanded_mask = mask[:, None, None].expand_as(Vh)
            expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
            
           # VhSinv  = torch.diag_embed(Sv)@Vhinv
           # mask = Sv <= 1e-5   
          #  Sv[mask] = 10e9    
           # VhS = (Vh@torch.diag_embed(1/Sv))
      
            # Divide the matrices by their determinant only where the determinant is non-zero
            Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
            Vhinv[expanded_mask] = 0.05*Vhinv[expanded_mask] * torch.abs(expanded_det_Vh[expanded_mask])
           # VhS[expanded_mask] = VhS[expanded_mask] /  expanded_det_Vh[expanded_mask]
            #VhSinv[expanded_mask] = VhSinv[expanded_mask] * expanded_det_Vh[expanded_mask]
            Vh=Vh.unsqueeze(1).unsqueeze(1)
            Vhinv2 = Vhinv.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,1,self.embed_dim,1,1 )
            Vhinv=Vhinv.unsqueeze(1).unsqueeze(1)
          #  VhS=VhS.unsqueeze(1).unsqueeze(1)
          #  VhSinv=VhSinv.unsqueeze(1).unsqueeze(1)
            y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
            y1 = torch.abs(x.unsqueeze(-2)@(Vhinv.transpose(-1,-2))).squeeze(-2)
      
            x2 = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim,1,1)
        

        weightASym  =  self.weightASym - self.weightASym.permute(0,2,1)  
        weightSym  =  self.weightSym + self.weightSym.permute(0,2,1)   
       # weightTU = torch.softmax( torch.triu(self.weightTU),dim=1)
        #weightTL = torch.softmax( torch.tril(self.weightTL),dim=1)
        weightTU =  torch.triu(self.weightTU)
        weightTL = torch.tril(self.weightTL)
       # y2 = self.proj_inputS1(torch.abs(x2@(weightSym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2))
       # y3 = self.proj_inputS2(torch.abs(x2@(weightASym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2))
        #y4 = self.proj_inputS3(torch.abs(x2@(weightTU@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2))
       # y5 = self.proj_inputS4(torch.abs(x2@(weightTL@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2))
        y2 = torch.abs(x2@(weightSym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        y3 = torch.abs(x2@(weightASym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        y4 = torch.abs(x2@(weightTU@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        y5 = torch.abs(x2@(weightTL@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)
        z = torch.cat([torch.selu(y2),torch.selu(y3),torch.selu(y4),torch.selu(y5)],dim=-1)
        #z = torch.selu(self.proj_inputS(torch.cat([torch.selu(y2),torch.selu(y3),torch.selu(y4),torch.selu(y5)],dim=-1)))
        x =  torch.cat([torch.selu(y),torch.selu(y1)],dim=-1)
            #print("x.shape", x.shape)
        x = torch.cat([torch.selu(self.proj_input(x)),z],dim=-1).reshape(bs, idim, idim , 64)
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)
        #print("x.shape",x.shape)
        x = self.finalPooling(x)
        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs      
    
class ResNetBase2lgatsmall(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[32,32,32]):#resnet tiny
        super(ResNetBase2lgatsmall, self).__init__(recurrent, num_inputs, hidden_size)
        self.embed_dim = 5 # channels[0] -8
        self.layer1 = self._make_layer(channels[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(2*num_inputs, 4)
        #self.proj_inputS = nn.Linear(self.embed_dim*num_inputs, channels[0]-8)
        self.proj_inputS1 = nn.Linear(self.embed_dim*num_inputs, 7)
        self.proj_inputS2 = nn.Linear(self.embed_dim*num_inputs, 7)
        self.proj_inputS3 = nn.Linear(self.embed_dim*num_inputs, 7)
        self.proj_inputS4 = nn.Linear(self.embed_dim*num_inputs, 7)
        
        #self.proj_input2 = nn.Linear(32, num_inputs)
        
        self.finalPooling = nn.MaxPool2d((2,2), stride=(2,2))

        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048//4, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        colors = 3
        
        self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightASym = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightASym, a=math.sqrt(5))
        self.weightTU = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightTU, a=math.sqrt(5))
        self.weightTL = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightTL, a=math.sqrt(5))
        

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    
    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
            Vhinv = torch.linalg.pinv(Vh)
            det_Vh = torch.det(Vh)

        
        return U,Sv.to(device),Vh.to(device),Vhinv.to(device),det_Vh.to(device)
    
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
    

        if idim ==1:
            #dummy
            x=x
        else:
            
            x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
            x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
            #print( x0.shape)
            #x = self.get_patches_flat(x,ps=self.patch_size)
            _,Sv,Vh, Vhinv ,det_Vh = self.unique_svd(x0)
    
            mask = det_Vh != 0
            # Expand the mask and determinant to have the same shape as Vh for broadcasting
            expanded_mask = mask[:, None, None].expand_as(Vh)
            expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)
            
           # VhSinv  = torch.diag_embed(Sv)@Vhinv
           # mask = Sv <= 1e-5   
          #  Sv[mask] = 10e9    
           # VhS = (Vh@torch.diag_embed(1/Sv))
      
            # Divide the matrices by their determinant only where the determinant is non-zero
            Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
            Vhinv[expanded_mask] = Vhinv[expanded_mask] * expanded_det_Vh[expanded_mask]
           # VhS[expanded_mask] = VhS[expanded_mask] /  expanded_det_Vh[expanded_mask]
            #VhSinv[expanded_mask] = VhSinv[expanded_mask] * expanded_det_Vh[expanded_mask]
            Vh=Vh.unsqueeze(1).unsqueeze(1)
            Vhinv2 = Vhinv.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,1,self.embed_dim,1,1 )
            Vhinv=Vhinv.unsqueeze(1).unsqueeze(1)
          #  VhS=VhS.unsqueeze(1).unsqueeze(1)
          #  VhSinv=VhSinv.unsqueeze(1).unsqueeze(1)
            y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
            y1 = torch.abs(x.unsqueeze(-2)@(Vhinv.transpose(-1,-2))).squeeze(-2)
      
            x2 = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim,1,1)
        

        weightASym  =  self.weightASym - self.weightASym.permute(0,2,1)  
        weightSym  =  self.weightSym + self.weightSym.permute(0,2,1)   
        weightTU = torch.softmax( torch.triu(self.weightTU),dim=1)
        weightTL = torch.softmax( torch.tril(self.weightTL),dim=1)
        y2 = self.proj_inputS1(torch.selu(torch.abs(x2@(weightSym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)))
        y3 = self.proj_inputS2(torch.selu(torch.abs(x2@(weightASym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)))
        y4 = self.proj_inputS3(torch.selu(torch.abs(x2@(weightTU@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)))
        y5 = self.proj_inputS4(torch.selu(torch.abs(x2@(weightTL@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)))
        z = torch.cat([y2,y3,y4,y5],dim=-1)
        #z = torch.selu(self.proj_inputS(torch.cat([torch.selu(y2),torch.selu(y3),torch.selu(y4),torch.selu(y5)],dim=-1)))
        x =  torch.cat([torch.selu(y),torch.selu(y1)],dim=-1)
            #print("x.shape", x.shape)
        x = torch.cat([torch.selu(self.proj_input(x)),torch.selu(z)],dim=-1).reshape(bs, idim, idim , 32)
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)
        #print("x.shape",x.shape)
        x = self.finalPooling(x)
        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs      
    
    
class ResNetBase2lgatsmall2(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[32,32,32]):#resnet tiny
        super(ResNetBase2lgatsmall2, self).__init__(recurrent, num_inputs, hidden_size)
        self.embed_dim = 5 # channels[0] -8
        self.layer1 = self._make_layer(channels[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(2*num_inputs, 4)
        #self.proj_inputS = nn.Linear(self.embed_dim*num_inputs, channels[0]-8)
        self.proj_inputS1 = nn.Linear(self.embed_dim*num_inputs, 7)
        self.proj_inputS2 = nn.Linear(self.embed_dim*num_inputs, 7)
        self.proj_inputS3 = nn.Linear(self.embed_dim*num_inputs, 7)
        self.proj_inputS4 = nn.Linear(self.embed_dim*num_inputs, 7)
        
        #self.proj_input2 = nn.Linear(32, num_inputs)
        
        self.finalPooling = nn.MaxPool2d((2,2), stride=(2,2))

        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)

        self.fc = init_relu_(nn.Linear(2048//4, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()
        
        colors = 3
        
        self.weightSym = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightSym, a=math.sqrt(5))
        self.weightASym = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightASym, a=math.sqrt(5))
        self.weightTU = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightTU, a=math.sqrt(5))
        self.weightTL = nn.Parameter(torch.Tensor(self.embed_dim, colors,colors))
        nn.init.kaiming_uniform_(self.weightTL, a=math.sqrt(5))
        

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    
    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
            Vhinv = torch.linalg.pinv(Vh)
            #det_Vh = torch.det(Vh)

        
        return U,Sv.to(device),Vh.to(device),Vhinv.to(device)
    
         
    
         
    def get_patches_flat(self, x, ps):
        bs,  h, w,c = x.size()    
        patches = x.unfold(1, ps,   ps).permute(0,1,4, 2, 3)
        patches = patches.unfold(3,  ps,  ps).permute(0, 1, 3, 2,5,4)  
        return patches.reshape((-1, ps**2,c)) #
    
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
    

        if idim ==1:
            #dummy
            x=x
        else:
            
            x0 = self.get_patches_flat(x,ps=self.patch_size)
            #print( x0.shape)
            x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
            #print( x0.shape)
            #x = self.get_patches_flat(x,ps=self.patch_size)
            _,Sv,Vh, Vhinv  = self.unique_svd(x0)
    

           
            Vh=Vh.unsqueeze(1).unsqueeze(1)
            Vhinv2 = Vhinv.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,1,self.embed_dim,1,1 )
            Vhinv=Vhinv.unsqueeze(1).unsqueeze(1)
          #  VhS=VhS.unsqueeze(1).unsqueeze(1)
          #  VhSinv=VhSinv.unsqueeze(1).unsqueeze(1)
            y = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
            y1 = torch.abs(x.unsqueeze(-2)@(Vhinv.transpose(-1,-2))).squeeze(-2)
      
            x2 = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,1,self.embed_dim,1,1)
        

        weightASym  =  self.weightASym - self.weightASym.permute(0,2,1)  
        weightSym  =  self.weightSym + self.weightSym.permute(0,2,1)   
        weightTU =  torch.triu(self.weightTU)
        weightTL = torch.tril(self.weightTL)
        y2 = self.proj_inputS1(torch.selu(torch.abs(x2@(weightSym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)))
        y3 = self.proj_inputS2(torch.selu(torch.abs(x2@(weightASym@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)))
        y4 = self.proj_inputS3(torch.selu(torch.abs(x2@(weightTU@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)))
        y5 = self.proj_inputS4(torch.selu(torch.abs(x2@(weightTL@(Vhinv2.transpose(-1,-2)))).squeeze(-2).flatten(-2)))
        z = torch.cat([y2,y3,y4,y5],dim=-1)
        #z = torch.selu(self.proj_inputS(torch.cat([torch.selu(y2),torch.selu(y3),torch.selu(y4),torch.selu(y5)],dim=-1)))
        x =  torch.cat([torch.selu(y),torch.selu(y1)],dim=-1)
            #print("x.shape", x.shape)
        x = torch.cat([torch.selu(self.proj_input(x)),torch.selu(z)],dim=-1).reshape(bs, idim, idim , 32)
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)
        #print("x.shape",x.shape)
        x = self.finalPooling(x)
        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs       
    
        
    
class ResNetBase2ll(NNBase):
    """
    Residual Network 
    """
    #def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
    #def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[8,16,32]): #resnet small
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, channels=[16,32,32]):#resnet tiny
        super(ResNetBase2ll, self).__init__(recurrent, num_inputs, hidden_size)

        self.layer1 = self._make_layer(4*num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.proj_input = nn.Linear(4*num_inputs, 4*num_inputs)
        #self.proj_input2 = nn.Linear(32, num_inputs)
        
        self.idim = 64
        self.patch_size = 8
        self.num_patches = (self.idim // self.patch_size)**2
        
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.norm1  = nn.LayerNorm(num_inputs)
        self.norm2  = nn.LayerNorm(num_inputs)
        
        self.pl = 5 #3
        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.patch_size_l = self.pl*self.patch_size # 3*self.patch_size 
        self.patch_idxs = self.create_patching_idxs(dim = self.idim , ps= self.patch_size_l  , ps2 = self.patch_size ) # take self.patch_size_l  =3 nieghbouring patchs verticaly and horizontally -> 9 in total
        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)
    
    def unique_svd(self,x):
        #python veriosn if old torch.svd, python verions new torch.linalg.svd
        device = x.device
        x = x.detach().cpu()
        
        with torch.no_grad():
            U,Sv,Vh = torch.svd(x)
            Vhinv = torch.linalg.pinv(Vh)
            det_Vh = torch.det(Vh)

        
        return U,Sv.to(device),Vh.to(device),Vhinv.to(device),det_Vh.to(device)
    
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
        padding = int((self.patch_size_l)//self.pl) #  int((self.patch_size_l)//3)
        y = F.pad(x.permute(0,3,1,2), (padding, padding, padding, padding), mode='constant', value=0)
        y = torch.index_select(y.flatten(-2), 2, self.patch_idxs.to(x.device))
        #print(y.shape)
        y =y.reshape(bs,c,self.num_patches ,self.patch_size_l,self.patch_size_l).permute(0,2,3,4,1)
        y = self.get_patches_flat(y.reshape(bs*self.num_patches ,self.patch_size_l,self.patch_size_l,c), ps=self.patch_size)
        y = y.reshape((bs, self.num_patches, self.pl**2,self.patch_size*self.patch_size, c)).mean(2)
        #print(y.shape)
        return y.reshape((bs*self.num_patches,self.patch_size*self.patch_size, c)) #
        
    def forward(self, inputs, rnn_hxs, masks):
        
        
        x = inputs.permute(0,2,3,1)
      #  print("x.shape",x.shape)
        bs, idim, idim , cs = x.size()  
        self.patch_size = 8
        self.num_patches = 64
    

        if idim ==1:
            #dummy
            x=x
        else:
            with torch.no_grad():
                x0 = self.get_patches_mini(x)
                #print( x0.shape)
                #x0 = x0.reshape(bs,self.num_patches,self.patch_size**2,cs).mean(1)
                #print( x0.shape)
                x = self.get_patches_flat(x,ps=self.patch_size)
                _,Sv,Vh, Vhinv ,det_Vh = self.unique_svd(x0)

                mask = det_Vh != 0
                # Expand the mask and determinant to have the same shape as Vh for broadcasting
                expanded_mask = mask[:, None, None].expand_as(Vh)
                expanded_det_Vh = det_Vh[:, None, None].expand_as(Vh)

                VhSinv  = torch.diag_embed(Sv)@Vhinv
                mask = Sv <= 1e-5   
                Sv[mask] = 10e9    
                VhS = (Vh@torch.diag_embed(1/Sv))

             
                Vh[expanded_mask] = Vh[expanded_mask] /  expanded_det_Vh[expanded_mask]
                Vhinv[expanded_mask] = Vhinv[expanded_mask] * expanded_det_Vh[expanded_mask]
                VhS[expanded_mask] = VhS[expanded_mask] /  expanded_det_Vh[expanded_mask]
                VhSinv[expanded_mask] = VhSinv[expanded_mask] * expanded_det_Vh[expanded_mask]
                Vh=Vh.unsqueeze(1)
                Vhinv=Vhinv.unsqueeze(1)
                VhS=VhS.unsqueeze(1)
                VhSinv=VhSinv.unsqueeze(1)
               # print("x.shape", x.shape)
                #print("Vh.shape", Vh.shape)
                y2 = torch.abs(x.unsqueeze(-2)@Vh).squeeze(-2) # the abs needed as VhS defined up to signs 
               # print("y2.shape", y2.shape)
                y3 = torch.abs(x.unsqueeze(-2)@(Vhinv.transpose(-1,-2))).squeeze(-2)
                y2a = torch.abs(x.unsqueeze(-2)@VhS).squeeze(-2) # the abs needed as VhS defined up to signs 
                y3a = torch.abs(x.unsqueeze(-2)@(VhSinv.transpose(-1,-2))).squeeze(-2)
    
                x = torch.cat([torch.selu(y2),torch.selu(y3),torch.selu(y2a),torch.selu(y3a)],dim=-1)
                #print("x.shape", x.shape)
                x = self.reconstruct_image(x.reshape((bs,self.num_patches,self.patch_size**2, 4*cs)), ps=self.patch_size, img_dim = self.idim)
            x = torch.selu(self.proj_input(x).reshape(bs, idim, idim , 4*cs)) 
            #x = x.reshape(bs, idim, idim , 4*cs) 

        x = self.layer1(x.permute(0,3,1,2))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
    
    
    


