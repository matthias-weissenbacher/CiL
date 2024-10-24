# CiL
Color-Invariant Layers for Generalization in image based RL, accepted to be published in TMLR.



#Our DMControl SAC code is based on: SVEA: Stabilized Q-Value Estimation under Data Augmentation
 [this repository](https://github.com/nicklashansen/svea-vit) for SVEA implemented using ConvNets, as well as additional baselines and test environments.
 And our PPO backbone is given by DrAC in [this repository](https://github.com/rraileanu/auto-drac).
 
 ## Setup SVEA and DrAC
 
 ## SVEA Modifications for CiL
  in CiL_SVEA we have added a modified version of the above repo, replace src directpry with our code (our src de snot conatin env folder, keep the latter from orginal repo.) 
  
  
   
 ## DrAC Modifications for CiL
  in CiL_SVEA we have added a modified version of the above repo, replace src directpry with our code (our src de snot conatin env folder, keep the latter from orginal repo.) 
  
  
 ## DrAC Modifications for CiL
  in Cil_DrAC we have added the backboe ResNet model with CiL (all versions, Cil*,Cil3,CiL5 etc. referred to as local and global). place the atached ResNet.py in /ucb_rl2_meta. in model.py add,
 ```
from ucb_rl2_meta.ResNet import ResNetBaseLocal2
from ucb_rl2_meta.ResNet import ResNetBaseGlobal
from ucb_rl2_meta.ResNet import ResNetBaseGlobal2
from ucb_rl2_meta.ResNet import ResNetBaseGlobal3
```

 the local CiL attention window can be changed by setting self.mult =1,3,5, (for Cil1,Cil3,CiL5)

 ## SVEA: Training CNN + CiL 

```
conda activate svea
python3 src/train.py --domain_name walker --task_name walk  --seed 1 --device_id 0 --batch_size 128 --run_name CNN_CiL --hidden_dim 64  --algorithm sac  
```


 ## DrAC: Training CNN + CiL 
 
 Here you need to replace in model.py line 140 set
 
 ```
 base = ResNetBaseLocal2 # or other model you want to test
```

Run as DrAC + crop.


