"""Independent checkpoint rollout; count catastrophic causes before environment reset."""
import json
import torch
from t800_amp_getup import ROOT,mdp
from t800_amp_getup.host_task import T800HoSTTask,host_cfg
from t800_amp_getup.cli import make_runner
from mjlab.rl import RslRlVecEnvWrapper
OUT=ROOT/'logs/catastrophic_4500';OUT.mkdir(exist_ok=True)
torch.manual_seed(219)
names=['nonfinite_joint_position','nonfinite_root_pose','nonfinite_root_velocity','nonfinite_joint_velocity','joint_speed_gt120','root_speed_gt12','root_height_below_minus015','root_height_above17','horizontal_distance_gt25']
counts=torch.zeros(9,device='cuda',dtype=torch.long);joint_counts=torch.zeros(25,device='cuda',dtype=torch.long);total_cat=0;total_done=0;term_counts={};peak_joint=0.;peak_root=0.;noise=None

def audited(env):
 global total_cat,peak_joint,peak_root
 d=mdp.robot(env).data;h=mdp.base_height(env)
 checks=torch.stack([~torch.isfinite(d.joint_pos).all(-1),~torch.isfinite(d.root_link_pose_w).all(-1),~torch.isfinite(d.root_link_vel_w).all(-1),~torch.isfinite(d.joint_vel).all(-1),d.joint_vel.abs().amax(-1)>120,d.root_link_lin_vel_w.norm(dim=-1)>12,h<-.15,h>1.7,(d.root_link_pos_w[:,:2]-env.reset_xy).norm(dim=-1)>2.5],-1)
 mask=checks.any(-1)
 counts.add_(checks.sum(0));joint_counts.add_((d.joint_vel.abs()>120).sum(0))
 total_cat+=int(mask.sum());peak_joint=max(peak_joint,float(d.joint_vel.abs().max()));peak_root=max(peak_root,float(d.root_link_lin_vel_w.norm(dim=-1).max()))
 assert torch.equal(mask,mdp.catastrophic(env))
 return mask
cfg=host_cfg(256);cfg.seed=219;cfg.terminations['catastrophic'].func=audited
env=T800HoSTTask(cfg,device='cuda:0');w=RslRlVecEnvWrapper(env,clip_actions=3.)
try:
 runner=make_runner(w,OUT);runner.load(str(ROOT/'logs/host_full_stand_hold/model_4500.pt'));runner.eval_mode()
 obs=w.get_observations()
 with torch.inference_mode():
  for step in range(700):
   x=obs['actor'] if 'actor' in obs else obs['policy']
   if runner.cfg['empirical_normalization']:x=runner.obs_normalizer(x)
   action=runner.alg.policy.act(x)
   noise=float(runner.alg.policy.action_std.mean())
   obs,_,done,info=w.step(action);total_done+=int(done.sum())
   # Returned termination arrays are captured in logs before resets; count via compute hook below is unnecessary for cause proportions.
 report={'checkpoint':'model_4500.pt','num_envs':256,'steps':700,'seed':219,'stochastic_actions':True,'mean_action_std':noise,'assist_fraction':env.assistance_fraction,'total_episode_ends':total_done,'catastrophic_ends':total_cat,'catastrophic_fraction_of_ends':total_cat/max(total_done,1),'cause_counts':dict(zip(names,counts.cpu().tolist())),'joint_speed_violation_counts':dict(zip(env.scene['robot'].joint_names,joint_counts.cpu().tolist())),'max_abs_joint_speed':peak_joint,'max_root_linear_speed':peak_root,'note':'Independent rollout, not reconstruction of live training. Causes may overlap; partial episodes at end are censored.'}
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:w.close()
