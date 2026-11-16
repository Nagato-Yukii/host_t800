"""GPU integration regression: force schedule, partial resets, terminal features, sampler."""
import json
import torch
from t800_amp_getup import ROOT,mdp
from t800_amp_getup.host_task import host_cfg,T800HoSTTask
from t800_amp_getup.robot import BODY_NAMES,ANCHOR
from t800_amp_getup.motion_loader import AMPLoader
from mjlab.rl import RslRlVecEnvWrapper

def main():
 env=T800HoSTTask(host_cfg(8),device='cuda:0');w=RslRlVecEnvWrapper(env,clip_actions=3.)
 try:
  for it,ratio in [(0,.8),(499,.8),(500,.7),(3500,.1),(3999,.1),(4000,0),(4500,0)]:
   env.set_training_iteration(it);w.reset();w.step(torch.zeros(8,25,device=env.device))
   actual=env.scene['robot'].data.body_external_force
   assert torch.allclose(actual[:,env.torso_id,2],torch.full((8,),ratio*env.robot_mass*9.81,device=env.device),atol=1e-3)
   assert actual[:,:,:2].abs().max()==0
  w.reset();before=env.scene['robot'].data.joint_pos.clone();ids=torch.tensor([1,3],device=env.device)
  env.reset(env_ids=ids);after=env.scene['robot'].data.joint_pos
  assert torch.equal(before[[0,2,4,5,6,7]],after[[0,2,4,5,6,7]])
  # Force a known failure and verify true pre-reset AMP observation survives reset.
  rob=env.scene['robot'];pose=rob.data.root_link_pose_w[[0]].clone();pose[:,2]+=3
  rob.write_root_link_pose_to_sim(pose,env_ids=torch.tensor([0],device=env.device))
  obs,reward,done,info=w.step(torch.zeros(8,25,device=env.device))
  assert done[0];assert reward[0]<-10
  assert not torch.equal(info['terminal_amp'][0],obs['amp'][0]);assert mdp.base_height(env)[0]<.6
  loader=AMPLoader(str(ROOT/'assets/motions'),BODY_NAMES,ANCHOR,rob.body_names,env.device)
  motion,idx=loader.sample_indices(20000);assert motion.unique().numel()==531
  assert ((idx-loader.offsets[motion]+1)<loader.lengths[motion]).all()
  # Same FK state / feature conventions, expert frame 0 placed into the scene.
  import numpy as np,mujoco as mj
  from t800_amp_getup.robot import get_spec
  import zipfile,io
  with zipfile.ZipFile(ROOT/'data/T800_get_up_only_531.zip') as z:
   name=loader.motion_names[0];q=np.load(io.BytesIO(z.read('get_up/'+name+'/motion.npz')))['qpos'][0]
  pose=torch.tensor(q[:7],device=env.device,dtype=torch.float32)[None];pose[:,:3]+=env.scene.env_origins[[0]]
  rob.write_root_link_pose_to_sim(pose,env_ids=torch.tensor([0],device=env.device));rob.write_joint_state_to_sim(torch.tensor(q[7:],device=env.device,dtype=torch.float32)[None],torch.zeros(1,25,device=env.device),env_ids=torch.tensor([0],device=env.device))
  env.sim.forward();actual=env.observation_manager.compute_group('amp')
  k=len(BODY_NAMES)*9;error=(actual[0,:k]-loader.features[0,:k]).abs().max().item();assert error<1e-4,error
  report={'force_boundaries':'pass','partial_reset_isolation':'pass','terminal_amp_pre_reset':'pass','all_531_motions_sampled':'pass','no_cross_clip_pairs':'pass','expert_live_pose_feature_max_error':error}
  (ROOT/'logs/integration.json').write_text(json.dumps(report,indent=2));print(report)
 finally:w.close()
if __name__=='__main__':main()
