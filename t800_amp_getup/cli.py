import argparse,json,time
from pathlib import Path
import torch
from . import ROOT
from .host_task import host_cfg,T800HoSTTask,assistance_at_iteration
from .robot import BODY_NAMES,ANCHOR
from .test80task import T800Test80Task
from .host_slow_task import T800HoSTSlowTask,slow_assistance_at_iteration

def make_runner(env,log_dir):
 from dataclasses import asdict
 from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg
 from .runner import AmpOnPolicyRunner
 cfg=g1_amp_ppo_runner_cfg();cfg.amp_motion_files=str(ROOT/'assets/motions');cfg.amp_body_names=BODY_NAMES;cfg.amp_anchor_name=ANCHOR
 cfg.min_normalized_std=[.05]*25;cfg.amp_reward_coef=.08;cfg.amp_task_reward_lerp=.75
 cfg.actor.distribution_cfg['init_std']=.35
 cfg.experiment_name='t800_amp_getup';cfg.save_interval=100;cfg.amp_discr_hidden_dims=[512,256];cfg.num_steps_per_env=24
 return AmpOnPolicyRunner(env,asdict(cfg),str(log_dir),device=env.device)

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['check','smoke','train','play']);p.add_argument('--num-envs',type=int,default=64);p.add_argument('--iterations',type=int,default=10000);p.add_argument('--steps',type=int,default=400);p.add_argument('--checkpoint');p.add_argument('--log-dir');p.add_argument('--assist-iteration',type=int,default=8000);p.add_argument('--seed',type=int,default=42)
 p.add_argument('--task',choices=['host','test80','host_slow'],default='host')
 p.add_argument('--resume-iteration',type=int,help='Override learning iteration after loading a checkpoint; preserve learned states.')
 args=p.parse_args();torch.manual_seed(args.seed)
 if args.resume_iteration is not None and (args.command!='train' or not args.checkpoint or args.resume_iteration<0):p.error('--resume-iteration requires train, --checkpoint and a nonnegative iteration')
 if args.command=='check':
  from .robot import get_spec,PD
  m=get_spec().compile();print('model',m.nq,m.nv,'mass',m.body_mass.sum(),'CUDA',torch.cuda.is_available());print('PD',json.dumps(PD,indent=2));print('schedule',[.8 if args.task=='test80' else (slow_assistance_at_iteration(i) if args.task=='host_slow' else assistance_at_iteration(i)) for i in range(0,5500,500)]);return
 cfg=host_cfg(args.num_envs);cfg.seed=args.seed
 task_class={'host':T800HoSTTask,'test80':T800Test80Task,'host_slow':T800HoSTSlowTask}[args.task]
 env=task_class(cfg,device='cuda:0')
 from mjlab.rl import RslRlVecEnvWrapper
 wrapped=RslRlVecEnvWrapper(env,clip_actions=3.)
 try:
  if args.command=='smoke':
   from . import mdp
   assert env.max_episode_length==350
   assert (mdp.base_height(env)<.6).all()
   families=env.reset_family.unique().tolist();initial=mdp.base_height(env).tolist();resets=0
   for i in range(args.steps):
    obs,rew,done,info=wrapped.step(torch.zeros(args.num_envs,25,device=env.device))
    assert all(torch.isfinite(v).all() for v in obs.values());assert torch.isfinite(rew).all();resets+=int(done.sum())
    assert (env.episode_length_buf<=350).all();assert torch.allclose(env.force[:,0,2],torch.full((args.num_envs,),.8*env.robot_mass*9.81,device=env.device))
   report={'steps':args.steps,'num_envs':args.num_envs,'resets':resets,'families':families,'initial_height_range':[min(initial),max(initial)],'max_episode_steps':env.max_episode_length,'force_N':env.force[0,0,2].item(),'reward_terms':env.reward_manager.active_terms,'status':'passed'}
   (ROOT/'logs'/(args.task+'_smoke.json' if args.task!='host' else 'smoke.json')).write_text(json.dumps(report,indent=2));print(report)
  else:
   log=Path(args.log_dir) if args.log_dir else ROOT/'logs'/((args.task+'_' if args.task!='host' else '')+time.strftime('%Y%m%d_%H%M%S'));log.mkdir(parents=True,exist_ok=True)
   runner=make_runner(wrapped,log)
   if args.checkpoint:runner.load(args.checkpoint)
   if args.resume_iteration is not None:
    original_iteration=runner.current_learning_iteration
    runner.current_learning_iteration=args.resume_iteration
    env.set_training_iteration(args.resume_iteration)
    wrapped.reset()
    provenance={'source_checkpoint':str(Path(args.checkpoint).resolve()),'source_iteration':original_iteration,'resume_iteration':args.resume_iteration,'assist_fraction':env.assistance_fraction,'task':args.task,'num_envs':args.num_envs,'additional_iterations':args.iterations,'action_rate_weight':cfg.rewards['action_rate'].weight,'action_rate_func':cfg.rewards['action_rate'].func.__name__}
    (log/'resume_provenance.json').write_text(json.dumps(provenance,indent=2))
    runner.logger_type=runner.cfg.get('logger','tensorboard')
    runner.save(str(log/f'model_{args.resume_iteration}.pt'))
    print('[RESUME]',json.dumps(provenance),flush=True)
   if args.command=='train':runner.learn(args.iterations,init_at_random_ep_len=False)
   else:
    if not args.checkpoint:raise ValueError('play requires --checkpoint')
    env.set_training_iteration(args.assist_iteration);policy=runner.get_inference_policy();obs=wrapped.get_observations()
    with torch.inference_mode():
     for _ in range(args.steps):obs,_,_,_=wrapped.step(policy(obs))
 finally:wrapped.close()
if __name__=='__main__':main()
