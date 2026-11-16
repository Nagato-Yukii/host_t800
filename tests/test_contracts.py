import unittest
from types import SimpleNamespace as S
import torch,numpy as np
from t800_amp_getup.host_task import assistance_at_iteration,host_cfg
from t800_amp_getup.robot import PD,JOINT_NAMES
from t800_amp_getup import ROOT,mdp

class Contracts(unittest.TestCase):
 def test_slow_curriculum(self):
  from t800_amp_getup.host_slow_task import slow_assistance_at_iteration,T800HoSTSlowTask
  from t800_amp_getup.host_task import T800HoSTTask
  for it,value in [(0,.8),(2999,.3),(3000,.2),(3499,.2),(3500,.15),(3999,.15),(4000,.1),(4499,.1),(4500,.05),(4999,.05),(5000,0),(10000,0)]:
   self.assertAlmostEqual(slow_assistance_at_iteration(it),value)
  with self.assertRaises(ValueError):slow_assistance_at_iteration(-1)
  obj=object.__new__(T800HoSTSlowTask)
  for it in [3000,3500,4000,4500,5000]:
   obj.set_training_iteration(it)
   self.assertEqual(obj.training_iteration,it)
   self.assertEqual(obj.assistance_fraction,slow_assistance_at_iteration(it))
  self.assertIs(T800HoSTSlowTask.step,T800HoSTTask.step)
  self.assertIs(T800HoSTSlowTask._reset_idx,T800HoSTTask._reset_idx)

 def test_schedule_boundaries(self):
  for stage in range(8):
   self.assertAlmostEqual(assistance_at_iteration(stage*500),(8-stage)/10)
   self.assertAlmostEqual(assistance_at_iteration(stage*500+499),(8-stage)/10)
  self.assertEqual(assistance_at_iteration(4000),0);self.assertEqual(assistance_at_iteration(20000),0)
 def test_pd_semantic_mapping(self):
  self.assertEqual((PD['J00_HIP_PITCH_L']['kp'],PD['J00_HIP_PITCH_L']['kd']),(180,5))
  self.assertEqual((PD['J20_SHOULDER_PITCH_R']['kp'],PD['J20_SHOULDER_PITCH_R']['kd']),(60,.3))
  self.assertEqual((PD['J27_HEAD_PITCH']['kp'],PD['J27_HEAD_PITCH']['kd']),(100,1))
 def test_bank_and_data(self):
  z=np.load(ROOT/'data/fallen_resets.npz');self.assertEqual(z['joint_names'].tolist(),list(JOINT_NAMES))
  self.assertTrue(np.isfinite(z['qpos']).all());self.assertTrue((z['qpos'][:,2]<.6).all());self.assertEqual(np.bincount(z['family']).tolist(),[128]*4)
  self.assertEqual(len(list((ROOT/'assets/motions').glob('*.npz'))),531)
 def test_timing(self):
  cfg=host_cfg();self.assertEqual(cfg.episode_length_s,7);self.assertTrue(cfg.is_finite_horizon);self.assertEqual(cfg.decimation*cfg.sim.mujoco.timestep,.02)
  self.assertIs(cfg.rewards['action_rate'].func,mdp.common.action_rate_l2)
  self.assertEqual(cfg.rewards['action_rate'].weight,-.1)
  self.assertFalse(cfg.rewards['action_rate'].params)
  self.assertEqual(set(cfg.terminations),{'time_out','catastrophic','height_at_2s','height_at_5s'})
  self.assertEqual(cfg.terminations['height_at_2s'].params,{'time_s':2.,'min_height':.4})
  self.assertEqual(cfg.terminations['height_at_5s'].params,{'time_s':5.,'min_height':.8})
 def test_height_checkpoints(self):
  heights=torch.tensor([.39,.4,.41,.79,.8,.81])
  pos=torch.zeros(6,3);pos[:,2]=heights
  class Scene(dict):env_origins=torch.zeros(6,3)
  env=S(scene=Scene(robot=S(data=S(root_link_pos_w=pos))),episode_length_buf=torch.zeros(6,dtype=torch.long),step_dt=.02)
  for step in [99,100,101,249,250,251,350]:
   env.episode_length_buf.fill_(step)
   self.assertEqual(mdp.height_checkpoint(env,2.,.4).tolist(),[True,False,False,False,False,False] if step==100 else [False]*6)
   self.assertEqual(mdp.height_checkpoint(env,5.,.8).tolist(),[True,True,True,True,False,False] if step==250 else [False]*6)
 def test_no_hover_or_supine_standing(self):
  n=4;pos=torch.zeros(n,3);pos[:,2]=1.
  bodies=torch.zeros(n,3,3);bodies[:,0,:]=torch.tensor([0,.1,.06]);bodies[:,1,:]=torch.tensor([0,-.1,.06]);bodies[:,2,2]=1.6
  gravity=torch.tensor([[0.,0.,-1.]]).repeat(n,1);gravity[2,2]=0
  found=torch.ones(n,1,1,dtype=torch.bool);found[1]=False
  sensors={s:S(data=S(found=found.clone(),force=torch.tensor([[[0.,0.,100.]]]).repeat(n,1,1))) for s in ['foot_L','foot_R']}
  d=S(root_link_pos_w=pos,body_link_pos_w=bodies,projected_gravity_b=gravity,root_link_lin_vel_w=torch.zeros(n,3),root_link_ang_vel_w=torch.zeros(n,3));d.root_link_lin_vel_w[3,0]=1
  env=S(scene={'robot':S(data=d),**sensors},foot_ids=[0,1],head_id=2,robot_mass=80,num_envs=n,episode_length_buf=torch.full((n,),300),step_dt=.02)
  class Scene(dict):env_origins=torch.zeros(n,3)
  env.scene=Scene(env.scene)
  self.assertEqual(mdp.standing(env).tolist(),[True,False,False,False])
  self.assertEqual(mdp.stand_hold(env).tolist(),[1,0,0,0])
  for step in [0, 100, 249, 250, 251, 350]:
   env.episode_length_buf.fill_(step)
   self.assertEqual(mdp.stand_hold(env).tolist(),[1,0,0,0])
   if step <= 250:self.assertTrue((mdp.stand_still(env)==0).all())
  env.episode_length_buf.fill_(300)
  self.assertTrue((mdp.body_height(env)==0).all())
if __name__=='__main__':unittest.main()
