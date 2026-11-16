"""Measure reset-relative root XY displacement, not accumulated path length."""
from pathlib import Path
import csv,io,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/getup_displacement';OUT.mkdir(exist_ok=True)

def measure(xy,fps):
 d=np.linalg.norm(xy-xy[0],axis=1);t=np.arange(len(xy))/fps;cross=np.flatnonzero(d>2.5)
 return {'duration_s':float(t[-1]),'max_xy_distance_m':float(d.max()),'end_xy_distance_m':float(d[-1]),'path_length_m':float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),'max_first7s_m':float(d[t<=7].max()),'first_over_2p5_s':float(t[cross[0]]) if len(cross) else None,'fraction_frames_over_2p5':float((d>2.5).mean())}
rows=[]
with zipfile.ZipFile(ROOT/'data/T800_get_up_only_531.zip') as ar:
 source=json.loads(ar.read('get_up_results.json'))
 for entry in source:
  with np.load(io.BytesIO(ar.read(entry['output']+'/motion.npz'))) as z:
   raw=measure(z['qpos'][:,:2],float(z['fps']))
  name=Path(entry['output']).name
  with np.load(ROOT/'assets/motions'/(name+'.npz')) as z:
   idx=z['body_names'].tolist().index('LINK_BASE');amp=measure(z['body_pos_w'][:,idx,:2],float(z['fps']))
  rows.append({'clip':name,**amp,'source_max_m':raw['max_xy_distance_m'],'source_first7s_max_m':raw['max_first7s_m'],'parent_quality':entry['parent_quality']})
with (OUT/'per_clip.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
v=np.array([r['max_xy_distance_m'] for r in rows]);v7=np.array([r['max_first7s_m'] for r in rows]);src=np.array([r['source_max_m'] for r in rows])
report={'definition':'max_t norm(root_xy[t] - root_xy[0]); LINK_BASE, ground plane, meters; each get-up segment starts at its own first frame','num_clips':len(rows),'amp_max_displacement_quantiles_m':dict(zip(['min','p25','median','p75','p90','p95','p99','max'],np.quantile(v,[0,.25,.5,.75,.9,.95,.99,1]).tolist())),'mean_max_displacement_m':float(v.mean()),'threshold_exceedance_clips':{str(th):int((v>th).sum()) for th in [.5,1.,1.5,2.,2.5,3.]} ,'over_2p5_first7s':int((v7>2.5).sum()),'over_2p5_fullclip':int((v>2.5).sum()),'source_over_2p5_fullclip':int((src>2.5).sum()),'max_source_amp_peak_difference_m':float(np.abs(src-v).max()),'top10':sorted(rows,key=lambda r:r['max_xy_distance_m'],reverse=True)[:10]}
(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
