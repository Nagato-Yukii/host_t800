"""Verify constant assistance in the simulator across schedule boundaries and resets."""
import json
import torch
from mjlab.rl import RslRlVecEnvWrapper
from t800_amp_getup import ROOT
from t800_amp_getup.host_task import T800HoSTTask, host_cfg
from t800_amp_getup.test80task import T800Test80Task


def main():
    for method in ('step', '_reset_idx', '_apply_assistance'):
        assert getattr(T800Test80Task, method) is getattr(T800HoSTTask, method)
    env = T800Test80Task(host_cfg(8), device='cuda:0')
    wrapped = RslRlVecEnvWrapper(env, clip_actions=3.)
    results = []
    try:
        for iteration in (0, 999, 1000, 7000, 7999, 8000, 10000, 20000):
            env.set_training_iteration(iteration)
            wrapped.reset()
            for _ in range(2):
                obs, reward, done, info = wrapped.step(torch.zeros(8, 25, device=env.device))
                force = env.scene['robot'].data.body_external_force
                expected = torch.full((8,), .8 * env.robot_mass * 9.81, device=env.device)
                assert torch.allclose(force[:, env.torso_id, 2], expected, atol=1e-3)
                assert force[:, :, :2].abs().max() == 0
                assert env.torque.abs().max() == 0
                assert env.assistance_fraction == .8
                assert env.training_iteration == iteration
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
            results.append({'iteration': iteration, 'fraction': env.assistance_fraction, 'force_N': force[0, env.torso_id, 2].item()})
        report = {'status': 'passed', 'num_envs': 8, 'inherited_step_reset_force': True, 'reset_and_step_checks': results}
        (ROOT / 'logs/test80_integration.json').write_text(json.dumps(report, indent=2))
        print(report)
    finally:
        wrapped.close()


if __name__ == '__main__':
    main()
