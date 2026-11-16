from pathlib import Path
import hashlib,json,subprocess,zipfile,importlib.metadata
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(ROOT/'data/T800_get_up_only_531.zip') as z:(ROOT/'data/SOURCE_LICENSE.md').write_bytes(z.read('SOURCE_LICENSE.md'))
def git(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
amp=Path('/home/ye/BeyondMimicExp/AMP_mjlab')
deps={'amp_mjlab':{'path':str(amp),'commit':git(amp)},'HoST':{'path':'vendor/HoST','commit':git(ROOT/'vendor/HoST')},'engineai_rl_lab':{'path':'vendor/engineai_rl_lab','commit':git(ROOT/'vendor/engineai_rl_lab')},'runtime':{n:importlib.metadata.version(n) for n in ['torch','mjlab','mujoco','mujoco-warp','warp-lang']},'upstream_files':{str(p.relative_to(amp)):sha(p) for p in [amp/'src/tasks/amp_loco/amp_env_cfg.py',amp/'src/tasks/amp_loco/mdp/observations.py',amp/'src/tasks/amp_loco/mdp/rewards.py',amp/'rsl_rl/algorithms/amp_ppo.py',amp/'rsl_rl/runners/amp_on_policy_runner.py']}}
(ROOT/'docs/dependencies.json').write_text(json.dumps(deps,indent=2))
e=EventAccumulator(str(ROOT/'logs/train_validated'));e.Reload();tags=e.Tags()['scalars'];assert len(e.Scalars('Curriculum/iteration'))==16
for tag in tags:
 assert all(__import__('math').isfinite(x.value) for x in e.Scalars(tag)),tag
assert all(x.value==0 for x in e.Scalars('Loss/skipped_non_finite_batches'))
r=EventAccumulator(str(ROOT/'logs/resume_validated'));r.Reload();assert [x.value for x in r.Scalars('Curriculum/iteration')]==[16,17]
report={'train_iterations':16,'train_num_envs':16,'train_control_transitions':6144,'all_logged_scalars_finite':True,'skipped_nonfinite_batches':0,'resume_iterations':[16,17],'final_next_iteration':18,'tensorboard_tags':tags,'integration':json.loads((ROOT/'logs/integration.json').read_text()),'base':json.loads((ROOT/'logs/base_check.json').read_text()),'smoke':json.loads((ROOT/'logs/smoke.json').read_text())}
(ROOT/'docs/validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['tensorboard_tags','smoke']},indent=2))
