import torch,json
from t800_amp_getup.base_task import base_cfg,T800BaseTask
from t800_amp_getup.robot import PD,JOINT_NAMES
from mjlab.rl import RslRlVecEnvWrapper
from t800_amp_getup import ROOT
env=T800BaseTask(base_cfg(4),device='cuda:0');w=RslRlVecEnvWrapper(env)
try:
 assert list(env.scene['robot'].joint_names)==list(JOINT_NAMES)
 m=env.sim.mj_model
 for i,n in enumerate(JOINT_NAMES):
  assert abs(m.actuator_gainprm[i,0]-PD[n]['kp'])<1e-5
  assert abs(m.actuator_biasprm[i,2]+PD[n]['kd'])<1e-5
 for _ in range(5):
  obs,r,d,info=w.step(torch.zeros(4,25,device=env.device));assert torch.isfinite(r).all()
 assert env.scene['robot'].data.body_external_force.abs().max()==0
 (ROOT/'logs/base_check.json').write_text(json.dumps({'base_task':'passed','PD_all_25_sim_actuators':'exact','base_external_force':0},indent=2));print('BASE CHECK PASSED')
finally:w.close()
