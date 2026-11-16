from pathlib import Path
import ast,json,shutil,hashlib,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'vendor/engineai_rl_lab/source/engineai_rl_lab/engineai_rl_lab/tasks/locomotion/amp/robots/t800.py'
constants={}
for node in ast.parse(source.read_text()).body:
 if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
  try:constants[node.targets[0].id]=ast.literal_eval(node.value)
  except (ValueError,TypeError):pass
src=Path('/home/ye/GMR/assets/engineai_t800/t800.xml');r=ET.parse(src).getroot()
asset=ET.SubElement(r,'asset');meshes=ROOT/'assets/t800/meshes';meshes.mkdir(exist_ok=True)
for b in r.findall('.//body'):
 name=b.get('name');stl=Path('/home/ye/Desktop/ckx_exp/analysis/g1_to_t800_trial/visual_meshes')/f'{name}.stl'
 if stl.exists():
  shutil.copy2(stl,meshes/stl.name);ET.SubElement(asset,'mesh',name=name,file=f'meshes/{stl.name}')
  ET.SubElement(b,'geom',name=name+'_visual',type='mesh',mesh=name,contype='0',conaffinity='0',group='2',density='0',rgba='.65 .72 .8 1')
 for i,g in enumerate(b.findall('geom')):
  if g.get('type')!='mesh':g.set('name',f'{name}_collision_{i}');g.set('group','3');g.set('contype','1');g.set('conaffinity','1');g.set('condim','3');g.set('friction','.8 .02 .002')
ET.ElementTree(r).write(ROOT/'assets/t800/t800.xml')
# Map gains by semantic joint suffix, NEVER reuse the changed right-arm numeric IDs.
params={}
for j in r.findall('.//joint'):
 name=j.get('name');semantic='_'.join(name.split('_')[1:]);kind=semantic.removesuffix('_L').removesuffix('_R')
 if kind in ['HIP_PITCH','KNEE_PITCH']:motor='Q300H_L';kp='STIFFNESS_Q300H_L';kd='DAMPING_Q300H_L'
 elif kind=='HIP_ROLL':motor='Q300H';kp='STIFFNESS_Q300H';kd='DAMPING_Q300H'
 elif kind=='HIP_YAW':motor='Q200H';kp='STIFFNESS_Q200H';kd='DAMPING_Q200H'
 elif kind=='TORSO_YAW':motor='Q200H';kp='STIFFNESS_TORSO';kd='DAMPING_TORSO'
 elif kind.startswith('ANKLE'):motor='Q50H';kp='STIFFNESS_ANKLE';kd='DAMPING_ANKLE'
 elif kind.startswith('HEAD'):motor='Q25H';kp='STIFFNESS_HEAD';kd='DAMPING_HEAD'
 else:motor='Q25H' if kind=='ELBOW_YAW' else 'Q50H';kp='STIFFNESS_'+kind;kd='DAMPING_ARM'
 default=constants.get('DEFAULT_Q_'+semantic,constants.get('DEFAULT_Q_'+kind,0.0))
 params[name]={'kp':constants[kp],'kd':constants[kd],'armature':constants['ARMATURE_'+motor],'effort':constants['EFFORT_LIMIT_'+motor],'velocity':constants['VELOCITY_LIMIT_'+motor],'default':default}
(ROOT/'assets/t800/pd.json').write_text(json.dumps(params,indent=2))
zip_src=Path('/home/ye/Desktop/ckx_exp/analysis/g1_to_t800_batch/split_actions/T800_get_up_only_531.zip')
shutil.copy2(zip_src,ROOT/'data'/zip_src.name)
(ROOT/'docs/input_provenance.json').write_text(json.dumps({'source_archive_sha256':hashlib.sha256(zip_src.read_bytes()).hexdigest(),'gmr_xml_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'pd_source':str(source.relative_to(ROOT)),'pd_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2))
print('Prepared T800 asset, semantic PD mapping, exact 105MB source archive')
