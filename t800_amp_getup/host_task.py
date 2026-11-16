"""HoST-inspired T800 task. Assistance schedule is intentionally fixed HERE."""
import numpy as np
import torch
from mjlab.managers.reward_manager import RewardTermCfg as R
from mjlab.managers.termination_manager import TerminationTermCfg as T
from mjlab.utils.lab_api.math import quat_mul,quat_apply
from .base_task import T800BaseTask,base_cfg
from . import ROOT,mdp

def assistance_at_iteration(iteration:int)->float:
 if iteration<0:raise ValueError('iteration must be nonnegative')
 return max(0,8-int(iteration)//500)/10

def host_cfg(num_envs=64):
 cfg=base_cfg(num_envs)
 cfg.rewards['action_rate']=R(func=mdp.common.action_rate_l2,weight=-.1)
 cfg.rewards.update(body_height=R(func=mdp.body_height,weight=3.),head_height=R(func=mdp.head_height,weight=1.),upright=R(func=mdp.orientation,weight=1.),stand_hold=R(func=mdp.stand_hold,weight=5.),stand_still=R(func=mdp.stand_still,weight=1.),success=R(func=mdp.success_bonus,weight=1.))
 cfg.terminations.update(height_at_2s=T(func=mdp.height_checkpoint,params={'time_s':2.,'min_height':.4}),height_at_5s=T(func=mdp.height_checkpoint,params={'time_s':5.,'min_height':.8}))
 return cfg

class T800HoSTTask(T800BaseTask):
 curriculum_version = "fixed_weight_80_to_0_each_500_v1"
 def __init__(self,cfg,**kwargs):
  super().__init__(cfg,**kwargs)
  z=np.load(ROOT/'data/fallen_resets.npz')
  assert z['joint_names'].tolist()==list(self.scene['robot'].joint_names)
  self.fallen_bank=torch.tensor(z['qpos'],device=self.device,dtype=torch.float32)
  self.bank_family=torch.tensor(z['family'],device=self.device)
  self.stable_steps=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
  self.unstable_steps=torch.zeros_like(self.stable_steps)
  self.hold_all=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
  self.reset_family=torch.zeros_like(self.stable_steps)
  self.force=torch.zeros(self.num_envs,1,3,device=self.device);self.torque=torch.zeros_like(self.force)
  self.terminal_amp=None
  self.set_training_iteration(0)
 def set_training_iteration(self,iteration):
  super().set_training_iteration(iteration);self.assistance_fraction=assistance_at_iteration(iteration)
 def _apply_assistance(self):
  self.force.zero_();self.force[:,:,2]=self.assistance_fraction*self.robot_mass*9.81
  self.scene['robot'].write_external_wrench_to_sim(self.force,self.torque,body_ids=[self.torso_id])
 def _reset_idx(self,env_ids=None):
  if env_ids is None:env_ids=torch.arange(self.num_envs,device=self.device)
  super()._reset_idx(env_ids)
  if not hasattr(self,'fallen_bank'):return
  # Uniform four families; randomly select pre-settled pose within family and yaw.
  family=torch.randint(0,4,(len(env_ids),),device=self.device)
  idx=4*torch.randint(0,len(self.fallen_bank)//4,(len(env_ids),),device=self.device)+family
  q=self.fallen_bank[idx].clone();yaw=(torch.rand(len(env_ids),device=self.device)*2-1)*torch.pi
  rot=torch.zeros(len(env_ids),4,device=self.device);rot[:,0]=torch.cos(yaw/2);rot[:,3]=torch.sin(yaw/2)
  q[:,:2]=0;q[:,3:7]=quat_mul(rot,q[:,3:7]);q[:,:3]+=self.scene.env_origins[env_ids]
  rob=self.scene['robot'];rob.write_root_link_pose_to_sim(q[:,:7],env_ids=env_ids)
  rob.write_root_link_velocity_to_sim(torch.zeros(len(env_ids),6,device=self.device),env_ids=env_ids)
  rob.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=env_ids)
  self.reset_xy[env_ids]=q[:,:2];self.reset_family[env_ids]=family
  self.stable_steps[env_ids]=0;self.unstable_steps[env_ids]=0;self.hold_all[env_ids]=True
  self._apply_assistance()
 def step(self,action):
  # Explicit variant of mjlab ManagerBasedRlEnv.step: fresh contact/state evaluation,
  # every-physics-step global wrench and true PRE-reset AMP observations.
  self.action_manager.process_action(action.to(self.device).clamp(-3,3))
  for _ in range(self.cfg.decimation):
   self._sim_step_counter+=1;self.action_manager.apply_action();self._apply_assistance()
   self.scene.write_data_to_sim();self.sim.step();self.scene.update(dt=self.physics_dt)
  self.episode_length_buf+=1;self.common_step_counter+=1
  self.sim.forward();self.sim.sense()
  stable=mdp.standing(self)
  self.stable_steps=torch.where(stable,self.stable_steps+1,0)
  self.unstable_steps=torch.where(stable,0,self.unstable_steps+1)
  self.hold_all &= (self.episode_length_buf*self.step_dt<=5)|stable
  self.reset_buf=self.termination_manager.compute()
  self.reset_terminated=self.termination_manager.terminated.clone();self.reset_time_outs=self.termination_manager.time_outs.clone()
  self.reward_buf=self.reward_manager.compute(dt=self.step_dt);self.metrics_manager.compute()
  amp=self.observation_manager.compute_group('amp',update_history=False).clone()
  metrics={f'Reward/{name}':self.reward_manager._step_reward[:,i].mean().item()*self.step_dt for i,name in enumerate(self.reward_manager.active_terms)}
  metrics.update({'Task/height':mdp.base_height(self).mean().item(),'Task/standing_fraction':stable.float().mean().item(),'Curriculum/assist_fraction':self.assistance_fraction,'Curriculum/force_N':self.force[0,0,2].item(),'Curriculum/iteration':self.training_iteration})
  ids=self.reset_buf.nonzero(as_tuple=False).flatten()
  if len(ids):
   success=self.reset_time_outs[ids]&self.hold_all[ids]&stable[ids]&~self.reset_terminated[ids]
   metrics['Task/success_rate']=success.float().mean().item();metrics['Task/episode_seconds']=(self.episode_length_buf[ids].float()*self.step_dt).mean().item()
   for name in self.termination_manager.active_terms:
    metrics['Termination/'+name]=self.termination_manager.get_term(name)[ids].float().mean().item()
   self._reset_idx(ids);self.scene.write_data_to_sim()
  self.sim.forward();self.sim.sense()
  self.command_manager.compute(dt=self.step_dt)
  self.obs_buf=self.observation_manager.compute(update_history=True)
  self.extras['terminal_amp']=amp
  self.extras['log']=metrics
  self.extras['episode']=metrics.copy()
  return self.obs_buf,self.reward_buf,self.reset_terminated,self.reset_time_outs,self.extras
