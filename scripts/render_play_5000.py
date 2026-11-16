import os
os.environ['MUJOCO_GL']='egl'
import json
import numpy as np
import mujoco as mj
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from t800_amp_getup import ROOT
from t800_amp_getup.robot import get_spec
O=ROOT/'logs/play_5000';z=np.load(O/'rollout.npz');q=z['qpos'];ends={e['env']:e for e in json.loads((O/'ends.json').read_text())}
s=get_spec();s.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_PLANE,size=[10,10,.1],rgba=[.24,.27,.30,1]);s.worldbody.add_light(pos=[0,-3,5],dir=[0,0,-1],diffuse=[.8,.8,.8]);m=s.compile();d=mj.MjData(m)
m.vis.quality.shadowsize=1024
r=mj.Renderer(m,height=340,width=480);opt=mj.MjvOption();opt.geomgroup[3]=0
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
cams=[]
for k in range(8):
 c=mj.MjvCamera();c.lookat[:]=[0,0,.8];c.distance=3.5;c.azimuth=135;c.elevation=-15;cams.append(c)
path=O/'T800_test80_ckpt5000_8env_1x.mp4'
with imageio.get_writer(str(path),fps=25,codec='libx264',macro_block_size=None,ffmpeg_params=['-crf','22','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart']) as w:
 # Three seconds of held last frames allow inspection of the deadline failures.
 for frame in range((len(q)+1)//2+75):
  step=min(frame*2,len(q)-1);im=Image.new('RGB',(1920,864),(17,22,29));draw=ImageDraw.Draw(im)
  draw.text((15,10),'T800 test80 | checkpoint 5000 | 80% weight assist | 8 random resets | 1x speed | final frame held 3s',font=font,fill='white')
  for k in range(8):
   j=min(step,ends[k]['step']-1);d.qpos[:]=q[j,k];mj.mj_forward(m,d)
   r.update_scene(d,camera=cams[k],scene_option=opt)
   x=k%4*480;y=44+k//4*410
   im.paste(Image.fromarray(r.render()),(x,y+25));draw=ImageDraw.Draw(im)
   draw.text((x+8,y),f'Env {k} | reset family {z["families"][k]} | t={(j+1)*.02:.2f}s',font=small,fill='white')
   fail=[['height','upright','feet contact','feet height','support box','lin speed','ang speed'][a] for a in np.where(~z['checks'][j,k])[0]]
   draw.text((x+8,y+365),f'h={z["height"][j,k]:.2f}m  stable={z["stable_steps"][j,k]}/10  v={z["lin"][j,k]:.2f}  w={z["ang"][j,k]:.2f}',font=small,fill='white')
   draw.text((x+8,y+385),'Fail: '+(', '.join(fail) or 'none this step'),font=small,fill=(255,180,110))
  if frame in [0,60,124]:im.save(O/f'frame_{frame}.jpg')
  w.append_data(np.asarray(im))
r.close();print(path)
