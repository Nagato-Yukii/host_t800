import json
import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import BuiltinPositionActuatorCfg
from . import ROOT
PD=json.loads((ROOT/'assets/t800/pd.json').read_text())
JOINT_NAMES=tuple(PD)
BODY_NAMES=('LINK_BASE','LINK_HIP_ROLL_L','LINK_KNEE_PITCH_L','LINK_ANKLE_ROLL_L','LINK_HIP_ROLL_R','LINK_KNEE_PITCH_R','LINK_ANKLE_ROLL_R','LINK_SHOULDER_ROLL_L','LINK_ELBOW_PITCH_L','LINK_WRIST_END_L','LINK_SHOULDER_ROLL_R','LINK_ELBOW_PITCH_R','LINK_WRIST_END_R')
ANCHOR='LINK_TORSO_YAW'
def get_spec():
 return mujoco.MjSpec.from_file(str(ROOT/'assets/t800/t800.xml'))
def robot_cfg():
 return EntityCfg(spec_fn=get_spec,sort_actuators=True,
  init_state=EntityCfg.InitialStateCfg(pos=(0,0,1.0),joint_pos={n:v['default'] for n,v in PD.items()}),
  articulation=EntityArticulationInfoCfg(soft_joint_pos_limit_factor=.95,actuators=tuple(
   BuiltinPositionActuatorCfg(target_names_expr=(n,),stiffness=p['kp'],damping=p['kd'],effort_limit=p['effort'],armature=p['armature']) for n,p in PD.items())))
