from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];UP=Path('/home/ye/BeyondMimicExp/AMP_mjlab/rsl_rl')
s=(UP/'algorithms/amp_ppo.py').read_text()
s=s.replace('expert_state, expert_next_state = sample_amp_expert','expert_state, expert_next_state = sample_amp_expert\n            raw_policy_state = policy_state.detach().clone()\n            raw_expert_state = expert_state.detach().clone()')
s=s.replace('update(policy_state.cpu().numpy())','update(raw_policy_state.cpu().numpy())').replace('update(expert_state.cpu().numpy())','update(raw_expert_state.cpu().numpy())')
(ROOT/'t800_amp_getup/amp_ppo.py').write_text(s)
s=(UP/'runners/amp_on_policy_runner.py').read_text().replace('from rsl_rl.algorithms import AMPPPO','from .amp_ppo import AMPPPO').replace('from rsl_rl.utils import AMPLoader, Normalizer, store_code_state','from rsl_rl.utils import Normalizer, store_code_state\nfrom .motion_loader import AMPLoader')
s=s.replace('if init_at_random_ep_len:\n',"if init_at_random_ep_len:\n            raise ValueError('HoST requires complete 0-5-7s episodes; random episode ages are prohibited')\n")
s=s.replace('for it in range(start_iter, tot_iter):\n            start = time.time()', 'for it in range(start_iter, tot_iter):\n            self.env.unwrapped.set_training_iteration(it)\n            start = time.time()')
s=s.replace('self.train_mode()  # switch', 'obs = self.obs_normalizer(obs)\n        privileged_obs = self.privileged_obs_normalizer(privileged_obs)\n        self.train_mode()  # switch')
a=s.index('                    # Account for terminal state transitions');b=s.index('                    amp_obs = torch.clone(next_amp_obs)',a)
s=s[:a]+'''                    # Actual terminal features, never reset observations or self-transitions.
                    next_amp_obs_with_term = infos["terminal_amp"].to(self.device)
                    task_reward = rewards.clone()
                    rewards, disc_value = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, task_reward, normalizer=self.alg.amp_normalizer
                    )
                    alpha = self.alg.discriminator.task_reward_lerp
                    amp_contribution = rewards - alpha * task_reward
                    infos.setdefault("episode", {}).update({
                        "Reward/amp_weighted": amp_contribution.mean().item(),
                        "Reward/task_weighted": (alpha*task_reward).mean().item(),
                        "Reward/total": rewards.mean().item(),
                        "AMP/discriminator": disc_value.mean().item(),
                    })
''' +s[b:]
s=s.replace('self.current_learning_iteration = it\n','self.current_learning_iteration = it + 1\n')
s=s.replace('"infos": infos,','"infos": infos,\n            "joint_names": list(self.env.unwrapped.scene["robot"].joint_names),\n            "curriculum_version": getattr(self.env.unwrapped, "curriculum_version", "fixed_weight_80_to_0_each_500_v1"),')
s=s.replace('loaded_dict = torch.load(path, weights_only=False)','loaded_dict = torch.load(path, map_location=self.device, weights_only=False)\n        if loaded_dict.get("joint_names") != list(self.env.unwrapped.scene["robot"].joint_names):\n            raise ValueError("checkpoint joint order mismatch")')
s=s.replace('self.current_learning_iteration = loaded_dict["iter"]','self.current_learning_iteration = loaded_dict["iter"]\n            self.env.unwrapped.set_training_iteration(self.current_learning_iteration)')
# All per-step log keys can vary when only some rollout steps reset.
s=s.replace('for key in locs["ep_infos"][0]:','for key in sorted(set().union(*(info.keys() for info in locs["ep_infos"]))):')
s=s.replace('if it % self.save_interval == 0:', 'if (it + 1) % self.save_interval == 0:').replace('f"model_{it}.pt"','f"model_{self.current_learning_iteration}.pt"')
(ROOT/'t800_amp_getup/runner.py').write_text(s)
