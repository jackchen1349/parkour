"""Pretraining script for morphology-agnostic parkour policy.

Phase 1: Spatial Domain Randomization (ξ ~ U(0.6, 1.4))
         + Discount Regularization (γ_reg = 0.98)

Trains thousands of robots with different leg lengths simultaneously
to produce a generalizable policy that can be quickly finetuned for
any specific morphology candidate.

Usage:
    python pretrain.py --task parkour --device cuda:0

Config overrides:
    --num_envs 4096          # reduce memory usage
    --max_iterations 10000   # train longer
    --run_name my_pretrain   # custom run name
"""

import os
import sys
import numpy as np
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def train(args):
    # Create log directory

    print("=" * 60)
    print("Phase 1: Pretraining with Spatial Domain Randomization")
    print("=" * 60)
    print(f"Morphology range: xi ~ U(0.6, 1.4)")
    print(f"Discount regularization: gamma = 0.98")
    print(f"Num environments: 6144")
    print(f"Device: {args.rl_device}")
    print("=" * 60)

    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # Force spatial domain randomization ON for pretraining
    env.cfg.domain_rand.spatial_domain_rand = True
    print(f"Spatial DR: {env.cfg.domain_rand.spatial_domain_rand}")
    print(f"Num buckets: {env.cfg.morphology.num_buckets}")

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )

    print(f"Log directory: {ppo_runner.log_dir}")
    print(f"Max iterations: {train_cfg.runner.max_iterations}")
    print(f"Num steps per env: {train_cfg.runner.num_steps_per_env}")
    print("=" * 60)

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )

    print("\nPretraining complete!")
    print(f"Model saved to: {ppo_runner.log_dir}")


if __name__ == '__main__':
    args = get_args()
    train(args)
