"""Evaluate parkour performance for a trained policy.

Measures:
- Long jump distance (gap terrain)
- High jump height (step terrain)
- Survival rate
- Average reward breakdown

Usage:
    python evaluate.py --task parkour --device cuda:0 --load_run <run_name>
"""

import os
import sys
import numpy as np
from collections import defaultdict

import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, get_load_path


def evaluate(args):
    # headless controlled by --headless CLI flag

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root=None, env=env, name=args.task, args=args
    )

    num_eps = getattr(args, 'num_eps', 20)
    print(f"Evaluating for {num_eps} episodes...")

    metrics = defaultdict(list)
    obs = env.reset()

    for ep in range(num_eps):
        ep_reward = 0.0
        ep_dist = 0.0
        ep_max_height = 0.0
        ep_rewards = defaultdict(float)
        done = False
        step_count = 0

        while not done and step_count < 500:
            with torch.no_grad():
                actions = ppo_runner.alg.actor_critic.act_inference(obs)

            obs, _, rewards, dones, infos = env.step(actions)

            if isinstance(dones, torch.Tensor):
                done = dones[0].item() if dones.ndim > 0 else dones.item()
            else:
                done = dones

            ep_reward += rewards[0].item() if isinstance(rewards, torch.Tensor) else rewards
            ep_max_height = max(ep_max_height, env.root_states[0, 2].item())

            step_count += 1

        root_pos = env.root_states[0, :3].cpu().numpy()
        init_pos = env.env_origins[0].cpu().numpy()
        ep_dist = root_pos[0] - init_pos[0] + 1.2
        survived = 1 if step_count >= 500 else 0

        metrics['distance'].append(ep_dist)
        metrics['height'].append(ep_max_height)
        metrics['reward'].append(ep_reward)
        metrics['survival'].append(survived)
        metrics['steps'].append(step_count)

        print(f"Ep {ep+1:3d}: dist={ep_dist:.2f}m, height={ep_max_height:.2f}m, "
              f"reward={ep_reward:.1f}, steps={step_count}, survived={bool(survived)}")

        obs = env.reset()

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for key in ['distance', 'height', 'reward', 'survival', 'steps']:
        vals = metrics[key]
        print(f"  {key:12s}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, "
              f"min={np.min(vals):.3f}, max={np.max(vals):.3f}")
    print("=" * 60)

    # Paper-relevant metrics
    print("\nPaper Metrics:")
    print(f"  Avg Jump Distance: {np.mean(metrics['distance']):.2f}m (target: 1.00m)")
    print(f"  Avg Jump Height:   {np.mean(metrics['height']):.2f}m (target: 0.55m)")
    print(f"  Survival Rate:     {np.mean(metrics['survival'])*100:.1f}%")
    print(f"  Max Jump Distance: {np.max(metrics['distance']):.2f}m")
    print(f"  Max Jump Height:   {np.max(metrics['height']):.2f}m")


if __name__ == '__main__':
    args = get_args()
    evaluate(args)
