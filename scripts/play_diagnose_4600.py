"""Read-only policy evaluation with pre-reset state capture for video and diagnosis."""
import json
import numpy as np
import torch
from t800_amp_getup import ROOT, mdp
from t800_amp_getup.host_task import host_cfg
from t800_amp_getup.host_task import T800HoSTTask
from t800_amp_getup.cli import make_runner
from mjlab.rl import RslRlVecEnvWrapper

OUT=ROOT/'logs/play_4600_noassist';OUT.mkdir(exist_ok=True)
torch.manual_seed(712)
env=T800HoSTTask(host_cfg(64),device='cuda:0');w=RslRlVecEnvWrapper(env,clip_actions=3.)
runner=make_runner(w,OUT)
runner.load(str(ROOT/'logs/host_full_stand_hold/model_4600.pt'))
env.assistance_fraction=0.
policy=runner.get_inference_policy();obs=w.get_observations()
active=torch.ones(64,dtype=torch.bool,device=env.device)
rows=[];poses=[];ends=[]
compute=env.termination_manager.compute

def capture():
    assert env.force.abs().max()==0
    done=compute()
    d=env.scene['robot'].data
    feet=d.body_link_pos_w[:,env.foot_ids];h=mdp.base_height(env)
    xy=d.root_link_pos_w[:,:2];lo=feet[:,:,:2].amin(1)-.12;hi=feet[:,:,:2].amax(1)+.12
    footz=(feet[:,:,2]-env.scene.env_origins[:,None,2]).abs().amax(-1)
    forces=[];hits=[]
    for side in ['L','R']:
        s=env.scene['foot_'+side].data
        hits.append(s.found.reshape(64,-1).any(-1))
        forces.append(s.force.reshape(64,-1,3).norm(dim=-1).sum(-1))
    lin=d.root_link_lin_vel_w.norm(dim=-1);ang=d.root_link_ang_vel_w.norm(dim=-1)
    checks=torch.stack([(h>.88)&(h<1.15),mdp.upright(env)>.92,mdp.feet_supported(env).all(-1),footz<.12,((xy>lo)&(xy<hi)).all(-1),lin<.5,ang<1.],-1)
    rows.append({'distance':(d.root_link_pos_w[:,:2]-env.reset_xy).norm(dim=-1).cpu().numpy(),'active':active.cpu().numpy().copy(),'checks':checks.cpu().numpy(),'height':h.cpu().numpy(),'upright':mdp.upright(env).cpu().numpy(),'feet_forces':torch.stack(forces,-1).cpu().numpy(),'feet_hits':torch.stack(hits,-1).cpu().numpy(),'footz':footz.cpu().numpy(),'lin':lin.cpu().numpy(),'ang':ang.cpu().numpy(),'stable_steps':env.stable_steps.cpu().numpy(),'done':done.cpu().numpy().copy()})
    pose=d.root_link_pose_w[:8].clone();pose[:,:3]-=env.scene.env_origins[:8]
    poses.append(torch.cat([pose,d.joint_pos[:8]],-1).cpu().numpy())
    for i in (done&active).nonzero().flatten().tolist():
        ends.append({'env':i,'step':int(env.episode_length_buf[i]),'family':int(env.reset_family[i]),'terms':[n for n in env.termination_manager.active_terms if env.termination_manager.get_term(n)[i]],'stable_steps':int(env.stable_steps[i])})
    active.logical_and_(~done)
    return done

env.termination_manager.compute=capture
families=env.reset_family.cpu().tolist()
try:
 with torch.inference_mode():
  for step in range(350):
   obs,_,_,_=w.step(policy(obs))
   if not active.any():break
 np.savez_compressed(OUT/'rollout.npz',qpos=np.stack(poses),families=np.array(families),**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
 (OUT/'ends.json').write_text(json.dumps(ends,indent=2))
 names=['height','upright','both_feet_support','feet_near_ground','root_inside_feet','linear_speed','angular_speed']
 summary={'checkpoint':'model_4600.pt','task':'host','assist_fraction':0.,'seed':712,'num_envs':64,'ends':ends,'criteria':names}
 deadline=[e['env'] for e in ends if e['step']==250]
 if deadline:
  checks=np.stack([r['checks'] for r in rows]);last=checks[249,deadline];window=checks[240:250,deadline]
  summary['deadline_envs']=len(deadline)
  summary['fail_at_250']={n:int((~last[:,i]).sum()) for i,n in enumerate(names)}
  summary['fail_any_last10']={n:int((~window[:,:,i]).any(0).sum()) for i,n in enumerate(names)}
  summary['final_ranges']={k:[float(np.min(rows[249][k][deadline])),float(np.median(rows[249][k][deadline])),float(np.max(rows[249][k][deadline]))] for k in ['height','upright','footz','lin','ang','stable_steps']}
 (OUT/'diagnosis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
finally:w.close()
