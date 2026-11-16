from pathlib import Path
import io,json,zipfile,sys
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from t800_amp_getup.robot import JOINT_NAMES,PD

def main():
 model=mj.MjModel.from_xml_path(str(ROOT/'assets/t800/t800.xml'));data=mj.MjData(model)
 names=[model.joint(i).name for i in range(1,model.njnt)];bodies=[model.body(i).name for i in range(1,model.nbody)]
 assert names==list(JOINT_NAMES)
 out=ROOT/'assets/motions';starts=[];manifest=[]
 with zipfile.ZipFile(ROOT/'data/T800_get_up_only_531.zip') as archive:
  rows=json.loads(archive.read('get_up_results.json'))
  for index,r in enumerate(rows):
   with np.load(io.BytesIO(archive.read(r['output']+'/motion.npz'))) as z:
    q=z['qpos'];assert z['joint_names'].tolist()==names
    old=np.arange(len(q))/float(z['fps']);t=np.arange(int(np.floor(old[-1]*50))+1)/50
    Q=np.stack([np.interp(t,old,q[:,i]) for i in range(q.shape[1])],axis=1)
    Q[:,3:7]=Slerp(old,Rotation.from_quat(q[:,[4,5,6,3]]))(t).as_quat()[:,[3,0,1,2]]
   poses=[];rots=[]
   for frame in Q:
    data.qpos[:]=frame;mj.mj_forward(model,data);poses.append(data.xpos[1:].copy());rots.append(data.xquat[1:].copy())
   pos=np.array(poses);quat=np.array(rots);lin=np.gradient(pos,.02,axis=0);ang=np.zeros_like(lin)
   for b in range(len(bodies)):
    rot=Rotation.from_quat(quat[:,b,[1,2,3,0]]);vel=(rot[1:]*rot[:-1].inv()).as_rotvec()/.02
    ang[:-1,b]=vel;ang[-1,b]=vel[-1]
   path=out/(Path(r['output']).name+'.npz')
   np.savez_compressed(path,fps=50,joint_names=names,body_names=bodies,joint_pos=Q[:,7:].astype('f'),joint_vel=np.gradient(Q[:,7:],.02,axis=0).astype('f'),body_pos_w=pos.astype('f'),body_quat_w=quat.astype('f'),body_lin_vel_w=lin.astype('f'),body_ang_vel_w=ang.astype('f'))
   starts.append(Q[0]);manifest.append({'file':path.name,'frames':len(Q),'parent_quality':r['parent_quality'],'segmentation_flags':r['segmentation_flags']})
   if index%100==0:print('motion',index+1,flush=True)
 np.save(ROOT/'data/source_start_qpos.npy',np.array(starts));(ROOT/'data/motion_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('converted',len(manifest),flush=True)
if __name__=='__main__':main()
