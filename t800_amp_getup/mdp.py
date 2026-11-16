"""Bounded task signals: ground contact is permitted during recovery, not a failure."""
import torch
from mjlab.envs import mdp as common
from src.tasks.amp_loco.mdp.rewards import self_collision_cost

def robot(env):return env.scene['robot']
def phase(env):
 t=env.episode_length_buf.float()*env.step_dt
 return torch.stack((t/7,(t>=5).float(),torch.full_like(t,env.assistance_fraction)),dim=-1)
def base_height(env):return robot(env).data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
def upright(env):return (-robot(env).data.projected_gravity_b[:,2]).clamp(0,1)
def feet_supported(env):
 checks=[]
 for side in ['L','R']:
  s=env.scene['foot_'+side].data
  hit=s.found.reshape(env.num_envs,-1).any(-1)
  force=s.force.reshape(env.num_envs,-1,3).norm(dim=-1).sum(-1)
  checks.append(hit & (force>env.robot_mass*9.81*.015))
 return torch.stack(checks,dim=-1)
def standing(env):
 d=robot(env).data;feet=d.body_link_pos_w[:,env.foot_ids];h=base_height(env)
 near=(feet[:,:,2]-env.scene.env_origins[:,None,2]).abs().amax(-1)<.12
 xy=d.root_link_pos_w[:,:2];lo=feet[:,:,:2].amin(1)-.12;hi=feet[:,:,:2].amax(1)+.12
 inside=((xy>lo)&(xy<hi)).all(-1)
 return (h>.88)&(h<1.15)&(upright(env)>.92)&feet_supported(env).all(-1)&near&inside&(d.root_link_lin_vel_w.norm(dim=-1)<.5)&(d.root_link_ang_vel_w.norm(dim=-1)<1.)
def body_height(env):
 # Absolute ground height AND upright: no reward for lifting feet/head while supine.
 h=base_height(env);return torch.exp(-((h-1.0)/.22)**2)*upright(env)*(env.episode_length_buf*env.step_dt<=5)
def head_height(env):
 d=robot(env).data;head=d.body_link_pos_w[:,env.head_id,2];feet=d.body_link_pos_w[:,env.foot_ids,2].mean(-1)
 return ((head-feet)/1.5).clamp(0,1)*upright(env)*(env.episode_length_buf*env.step_dt<=5)
def orientation(env):
 return upright(env)*((base_height(env)-.3)/.6).clamp(0,1)*(env.episode_length_buf*env.step_dt<=5)
def stand_hold(env):return standing(env).float()  # Reward stable standing throughout the episode.
def stand_still(env):
 d=robot(env).data;speed=d.root_link_lin_vel_w.square().sum(-1)+.25*d.root_link_ang_vel_w.square().sum(-1)
 return torch.exp(-5*speed)*standing(env)*(env.episode_length_buf*env.step_dt>5)
def energy(env):
 d=robot(env).data
 return (d.actuator_force*d.joint_vel).abs().sum(-1).clamp(max=20000)
def bounded_common(env,term,cap):return getattr(common,term)(env).clamp(0,cap)
def bounded_self_collision(env):return self_collision_cost(env,'self_collision',10.).clamp(0,4)
def terminal_cost(env):
 # Early exits must not avoid future costs; timeouts are NOT penalized as failures.
 t=env.episode_length_buf*env.step_dt
 return env.termination_manager.terminated.float()*(10+8*(7-t).clamp_min(0))/env.step_dt

def catastrophic(env):
 d=robot(env).data;h=base_height(env)
 bad=(~torch.isfinite(d.joint_pos).all(-1))|(~torch.isfinite(d.root_link_pose_w).all(-1))|(~torch.isfinite(d.root_link_vel_w).all(-1))|(~torch.isfinite(d.joint_vel).all(-1))
 return bad|(d.joint_vel.abs().amax(-1)>120)|(d.root_link_lin_vel_w.norm(dim=-1)>12)|(h<-.15)|(h>1.7)|((d.root_link_pos_w[:,:2]-env.reset_xy).norm(dim=-1)>2.5)
def height_checkpoint(env, time_s, min_height):
 # One-shot check at the specified episode age, using root height above terrain.
 at_checkpoint=env.episode_length_buf==round(time_s/env.step_dt)
 return at_checkpoint & (base_height(env)<min_height)
def success_bonus(env):
 end=env.episode_length_buf>=env.max_episode_length
 return (end & env.hold_all & standing(env)).float()*5/env.step_dt
