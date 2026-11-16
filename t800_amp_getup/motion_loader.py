"""Vectorized upstream AMP features; uniform sampling across ALL motions, 50Hz pairs."""
from pathlib import Path
import numpy as np
import torch
from mjlab.utils.lab_api.math import subtract_frame_transforms,matrix_from_quat,quat_apply_inverse

class AMPLoader:
 def __init__(self,motion_file,body_names,anchor_name,all_body_names,device='cuda:0'):
  self.device=device;features=[];lengths=[];self.motion_names=[]
  ids=[list(all_body_names).index(n) for n in body_names];anchor=list(all_body_names).index(anchor_name)
  for path in sorted(Path(motion_file).glob('*.npz')):
   with np.load(path) as z:
    assert z['body_names'].tolist()==list(all_body_names),(path,'body order mismatch')
    assert float(z['fps'])==50
    p=torch.as_tensor(z['body_pos_w'],device=device);q=torch.as_tensor(z['body_quat_w'],device=device)
    v=torch.as_tensor(z['body_lin_vel_w'],device=device);w=torch.as_tensor(z['body_ang_vel_w'],device=device)
   F=len(p);B=len(ids)
   pb,qb=subtract_frame_transforms(p[:,anchor,None].expand(-1,B,-1),q[:,anchor,None].expand(-1,B,-1),p[:,ids],q[:,ids])
   feat=torch.cat((pb.reshape(F,-1),matrix_from_quat(qb)[...,:,:2].reshape(F,-1),quat_apply_inverse(q[:,ids],v[:,ids]).reshape(F,-1),quat_apply_inverse(q[:,ids],w[:,ids]).reshape(F,-1)),dim=-1)
   assert torch.isfinite(feat).all();features.append(feat);lengths.append(F);self.motion_names.append(path.stem)
  self.features=torch.cat(features);self.lengths=torch.tensor(lengths,device=device,dtype=torch.long)
  self.offsets=torch.cat((torch.zeros(1,device=device,dtype=torch.long),self.lengths.cumsum(0)[:-1]))
  self.observation_dim=self.features.shape[-1];self.num_motions=len(lengths)
  print(f'[AMP] {self.num_motions} motions / {len(self.features)} frames / {self.observation_dim} features; uniform per-motion, 50Hz')
 def sample_indices(self,size):
  motion=torch.randint(self.num_motions,(size,),device=self.device)
  frame=(torch.rand(size,device=self.device)*(self.lengths[motion]-1)).long()
  return motion,self.offsets[motion]+frame
 def feed_forward_generator(self,num_mini_batch,mini_batch_size):
  for _ in range(num_mini_batch):
   _,idx=self.sample_indices(mini_batch_size)
   yield self.features[idx],self.features[idx+1]
