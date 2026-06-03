"""Pretraining script for morphology-agnostic parkour policy.

Phase 1: Spatial Domain Randomization (ξ ~ U(0.6, 1.4))
         + Discount Regularization (γ_reg = 0.98)

Usage:
    python pretrain.py --task parkour --device cuda:0
"""

import numpy as np

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def pretrain(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env.cfg.domain_rand.spatial_domain_rand = True

    print("=" * 60)
    print("Phase 1: Pretraining with Spatial Domain Randomization")
    print("=" * 60)
    print(f"Morphology range: xi ~ U({env.cfg.morphology.scaling_range[0]}, {env.cfg.morphology.scaling_range[1]})")
    print(f"Num environments: {env.cfg.env.num_envs}")
    print(f"Num buckets:       {env.cfg.morphology.num_buckets}")
    print(f"Spatial DR:        {env.cfg.domain_rand.spatial_domain_rand}")
    print(f"Obs noise:         {env.cfg.noise.add_noise}")
    print(f"Device:            {args.rl_device}")

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )

    print(f"Log directory:     {ppo_runner.log_dir}")
    print(f"Max iterations:    {train_cfg.runner.max_iterations}")
    print(f"Num steps/env:     {train_cfg.runner.num_steps_per_env}")
    print(f"Discount gamma:    {train_cfg.algorithm.gamma}")
    print("=" * 60)

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )

    print("\nPretraining complete!")
    print(f"Model saved to: {ppo_runner.log_dir}")


if __name__ == '__main__':
    args = get_args()
    pretrain(args)
