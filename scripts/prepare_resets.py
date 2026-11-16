from pathlib import Path
import sys,json,xml.etree.ElementTree as ET
import numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from t800_amp_getup.robot import PD

def lowest(m,d):
 low=10.
 for g in range(m.ngeom):
  if m.geom_contype[g]==0 or m.geom_type[g]==mj.mjtGeom.mjGEOM_PLANE:continue
  s=m.geom_size[g];R=d.geom_xmat[g].reshape(3,3)
  ext=s[0] if m.geom_type[g]==mj.mjtGeom.mjGEOM_SPHERE else np.dot(abs(R[2]),s)
  low=min(low,d.geom_xpos[g,2]-ext)
 return low

def main():
 r=ET.parse(ROOT/'assets/t800/t800.xml').getroot()
 # Load from original directory so relative mesh paths continue to work.
 ET.SubElement(r.find('worldbody'),'geom',name='ground',type='plane',size='10 10 .1',contype='1',conaffinity='1',friction='.8 .02 .002')
 path=ROOT/'assets/t800/reset_scene.xml';ET.ElementTree(r).write(path)
 m=mj.MjModel.from_xml_path(str(path));m.opt.timestep=.005;m.dof_damping[6:]=1.;m.dof_armature[6:]=[p['armature'] for p in PD.values()];d=mj.MjData(m)
 starts=np.load(ROOT/'data/source_start_qpos.npy');rng=np.random.default_rng(42);bank=[];labels=[];attempts=0;target=512
 while len(bank)<target and attempts<10000:
  family=len(bank)%4;attempts+=1;mj.mj_resetData(m,d)
  # HoST-style randomized lying orientation and joints, pre-settled OFF episode clock.
  eulers=[(0,np.pi/2,0),(0,-np.pi/2,0),(np.pi/2,0,0),(-np.pi/2,0,0)]
  q=starts[rng.integers(len(starts))].copy() if rng.random()<.5 else np.r_[np.zeros(3),[1,0,0,0],[p['default'] for p in PD.values()]]
  q[:2]=0
  q[3:7]=Rotation.from_euler('xyz',np.array(eulers[family])+rng.normal(0,.12,3)).as_quat()[[3,0,1,2]]
  q[7:]=np.clip(q[7:]+rng.uniform(-.08,.08,25),m.jnt_range[1:,0]+.01,m.jnt_range[1:,1]-.01)
  q[2]=1.;d.qpos[:]=q;mj.mj_forward(m,d);d.qpos[2]+=.02-lowest(m,d)
  for _ in range(100):mj.mj_step(m,d)
  mj.mj_forward(m,d)
  if not np.isfinite(d.qpos).all() or d.qpos[2]>.55:continue
  if any(c.dist<-.008 for c in d.contact):continue
  q=d.qpos.copy();q[2]+=.002-lowest(m,d)
  bank.append(q);labels.append(family)
 if len(bank)!=target:raise RuntimeError(f'Only {len(bank)} valid fallen poses')
 np.savez_compressed(ROOT/'data/fallen_resets.npz',qpos=np.array(bank),family=np.array(labels),joint_names=list(PD))
 (ROOT/'data/reset_report.json').write_text(json.dumps({'poses':len(bank),'families':['prone','supine','left_side','right_side'],'counts':np.bincount(labels).tolist(),'attempts':attempts,'max_root_height':float(np.max(np.array(bank)[:,2])),'settle_seconds':.5,'max_penetration_m':.008},indent=2))
 print('reset bank',len(bank),'attempts',attempts,flush=True)
if __name__=='__main__':main()
