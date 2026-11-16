"""Robot base task reuses AMP_mjlab managers, observations and general regularizers."""
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg as R
from mjlab.managers.termination_manager import TerminationTermCfg as T
from mjlab.envs import mdp as common
from mjlab.sensor import ContactSensorCfg,ContactMatch
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
from .robot import robot_cfg,BODY_NAMES,ANCHOR,JOINT_NAMES
from . import mdp

def base_cfg(num_envs=64):
 cfg=make_amp_env_cfg();cfg.scene.num_envs=num_envs;cfg.scene.extent=4.
 cfg.scene.entities={'robot':robot_cfg()};cfg.scene.terrain.terrain_type='plane';cfg.scene.terrain.terrain_generator=None
 cfg.commands={};cfg.curriculum={};cfg.events={};cfg.metrics={}
 cfg.scene.sensors=tuple(ContactSensorCfg(name='foot_'+s,primary=ContactMatch(mode='body',pattern='LINK_ANKLE_ROLL_'+s,entity='robot'),secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='netforce',num_slots=1) for s in ['L','R'])+(
  ContactSensorCfg(name='self_collision',primary=ContactMatch(mode='subtree',pattern='LINK_BASE',entity='robot'),secondary=ContactMatch(mode='subtree',pattern='LINK_BASE',entity='robot'),fields=('found','force'),reduce='netforce',num_slots=1,history_length=4),)
 for group in ['actor','critic']:
  terms=cfg.observations[group].terms
  for n in ['command','height_scan']:terms.pop(n,None)
  terms['base_ang_vel']=ObservationTermCfg(func=common.base_ang_vel)
  terms['phase']=ObservationTermCfg(func=mdp.phase)
 cfg.observations['critic'].terms['base_lin_vel']=ObservationTermCfg(func=common.base_lin_vel)
 cfg.observations['critic'].terms['height']=ObservationTermCfg(func=lambda env:mdp.base_height(env)[:,None])
 for group in ['critic','amp']:
  for term in cfg.observations[group].terms.values():
   if 'anchor_cfg' in term.params:term.params['anchor_cfg']=SceneEntityCfg('robot',body_names=(ANCHOR,),preserve_order=True)
   if 'body_cfg' in term.params:term.params['body_cfg']=SceneEntityCfg('robot',body_names=BODY_NAMES,preserve_order=True)
 cfg.observations['amp'].history_length=0  # live terminal features, not the previous history slot
 cfg.actions['joint_pos'].scale={n:(.15 if 'HEAD' in n else .75) for n in JOINT_NAMES}
 cfg.rewards={
  'joint_acc':R(func=mdp.bounded_common,weight=-2.5e-7,params={'term':'joint_acc_l2','cap':2e6}),
  'joint_limits':R(func=mdp.bounded_common,weight=-10.,params={'term':'joint_pos_limits','cap':.1}),
  'action_rate':R(func=mdp.bounded_common,weight=-.01,params={'term':'action_rate_l2','cap':50.}),
  'torques':R(func=mdp.bounded_common,weight=-2.5e-6,params={'term':'joint_torques_l2','cap':2e5}),
  'joint_velocity':R(func=mdp.bounded_common,weight=-.001,params={'term':'joint_vel_l2','cap':500.}),
  'energy':R(func=mdp.energy,weight=-2.5e-5),
  'self_collision':R(func=mdp.bounded_self_collision,weight=-.1),
  'termination':R(func=mdp.terminal_cost,weight=-1.),
 }
 cfg.terminations={'time_out':T(func=common.time_out,time_out=True),'catastrophic':T(func=mdp.catastrophic)}
 cfg.episode_length_s=7.;cfg.is_finite_horizon=True;cfg.decimation=4;cfg.sim.mujoco.timestep=.005
 cfg.sim.mujoco.iterations=20;cfg.sim.mujoco.ls_iterations=20;cfg.sim.mujoco.ccd_iterations=50
 cfg.sim.njmax=1500;cfg.sim.nconmax=128;cfg.sim.contact_sensor_maxmatch=256
 cfg.viewer.entity_name='robot';cfg.viewer.body_name=ANCHOR
 return cfg

class T800BaseTask(ManagerBasedRlEnv):
 def __init__(self,cfg,**kwargs):
  self.training_iteration=0;self.assistance_fraction=0.
  super().__init__(cfg,**kwargs)
  import torch
  rob=self.scene['robot'];self.foot_ids=[rob.body_names.index('LINK_ANKLE_ROLL_'+s) for s in ['L','R']]
  self.head_id=rob.body_names.index('LINK_HEAD_YAW');self.torso_id=rob.body_names.index(ANCHOR)
  self.robot_mass=float(self.sim.mj_model.body_mass[rob.indexing.body_ids.cpu().numpy()].sum())
  self.reset_xy=torch.zeros(self.num_envs,2,device=self.device)
 def set_training_iteration(self,iteration):self.training_iteration=int(iteration)

 def _reset_idx(self,env_ids=None):
  super()._reset_idx(env_ids)
  if hasattr(self,'reset_xy'):
   ids=slice(None) if env_ids is None else env_ids
   self.reset_xy[ids]=self.scene.env_origins[ids,:2]
