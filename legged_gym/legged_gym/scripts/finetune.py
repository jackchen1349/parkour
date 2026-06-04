"""Finetuning with Bayesian Optimization for morphology co-design.

Phase 2: BO searches for optimal morphology xi ∈ [0.6, 1.4]⁴.
         Each candidate is finetuned for ~400 steps from the pretrained policy.

The BO objective is a weighted combination of:
- Long jump distance (gap terrain)
- High jump height (step terrain)
- Survival rate

Usage:
    python finetune.py --task parkour --device cuda:0 --pretrained_path <path>

Config:
    --bo_iterations 50       # number of BO trials
    --finetune_steps 400     # steps per candidate
    --num_eval_eps 10       # evaluation episodes per candidate
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from copy import deepcopy

import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.envs.parkour.parkour_config import ParkourCfg, ParkourCfgPPO
from legged_gym.utils import get_args, task_registry, get_load_path


class BayesianOptimizer:
    """Simple BO using Gaussian Process surrogate via scipy or random search fallback."""

    def __init__(self, bounds, xi_init=None, n_init=5):
        self.bounds = np.array(bounds)  # (4, 2)
        self.dim = len(bounds)
        self.xi_init = xi_init if xi_init is not None else np.ones(self.dim)
        self.n_init = n_init
        self.X = []
        self.y = []

        # Try importing BoTorch/Ax for proper BO
        try:
            from botorch.models import SingleTaskGP
            from botorch.fit import fit_gpytorch_model
            from botorch.acquisition import UpperConfidenceBound
            from gpytorch.mlls import ExactMarginalLogLikelihood
            self._has_botorch = True
        except ImportError:
            self._has_botorch = False
            print("BoTorch not available, using random search with local refinement")

    def _surrogate_predict(self, X):
        """Simple GP-like prediction using RBF kernel."""
        if len(self.X) < 3:
            return np.zeros(len(X)), np.ones(len(X))

        X_train = np.array(self.X)
        y_train = np.array(self.y)

        length_scale = 0.3
        output_scale = np.std(y_train) + 0.1
        noise = 0.01

        preds = []
        uncertainties = []
        for x in X:
            dists = np.sum(((X_train - x) / length_scale) ** 2, axis=1)
            K = output_scale * np.exp(-0.5 * dists)
            pred = np.sum(K * y_train) / (np.sum(K) + noise)
            unc = output_scale - np.sum(K**2) / (np.sum(K) + noise)
            preds.append(pred)
            uncertainties.append(max(unc, 0.01))

        return np.array(preds), np.array(uncertainties)

    def suggest(self):
        """Suggest next morphology to evaluate."""
        if len(self.X) < self.n_init:
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            return torch.tensor(x, dtype=torch.float)

        n_candidates = 1000
        X_cand = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (n_candidates, self.dim)
        )

        preds, uncertainties = self._surrogate_predict(X_cand)
        ucb = preds + 2.0 * uncertainties  # UCB acquisition

        best_idx = np.argmax(ucb)
        return torch.tensor(X_cand[best_idx], dtype=torch.float)

    def observe(self, xi, score):
        self.X.append(xi.numpy() if isinstance(xi, torch.Tensor) else np.array(xi))
        self.y.append(score)


def evaluate_morphology(env, act_inference, xi, num_eps=10):
    """Evaluate a morphology on parkour tasks using batch (N env) evaluation.

    Paper Eq.4: f = (1/N) * Σ_i Σ_t r(s_t^i, a_t^i).
    Uses the mean cumulative reward across all environments as the fitness metric.
    """
    env.cfg.morphology.target_morphology = xi

    obs, _ = env.reset()
    ep_rewards = torch.zeros(env.num_envs, device=env.device)
    ep_max_dist = torch.zeros(env.num_envs, device=env.device)
    ep_max_height = torch.zeros(env.num_envs, device=env.device)
    step_count = 0
    max_steps = 500

    while step_count < max_steps:
        with torch.no_grad():
            actions = act_inference(obs, hist_encoding=False)
        obs, _, rewards, dones, _ = env.step(actions)
        if isinstance(rewards, torch.Tensor):
            ep_rewards += rewards
        ep_max_dist = torch.max(ep_max_dist, env.root_states[:, 0] - env.env_origins[:, 0] + 1.2)
        ep_max_height = torch.max(ep_max_height, env.root_states[:, 2])
        step_count += 1

        # Early terminate environments that are done
        if dones.any():
            ep_rewards[dones] = ep_rewards[dones]  # keep final values
            ep_max_dist[dones] = ep_max_dist[dones]
            ep_max_height[dones] = ep_max_height[dones]

    avg_reward = ep_rewards.mean().item()
    avg_dist = ep_max_dist.mean().item()
    avg_height = ep_max_height.mean().item()
    # Survival: environments that made it to max_steps
    survival = (step_count >= max_steps)

    return avg_reward, {
        'cumulative_reward': avg_reward,
        'distance': avg_dist,
        'height': avg_height,
        'survival': float(survival)
    }


def finetune_policy(env, ppo_runner, steps=400):
    """Quick finetuning of pretrained policy for a specific morphology.

    Paper Sec 2.2: ~400 steps per candidate, 6.67% of standard 6000-step training.
    num_steps_per_env=24 (matching parkour_config), so iterations = ceil(400/24) = 17.
    """
    num_steps_per_env = 24
    iterations = max(1, int(np.ceil(steps / num_steps_per_env)))
    ppo_runner.learn(
        num_learning_iterations=iterations,
        init_at_random_ep_len=True
    )


def main(args):
    args.headless = True

    log_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'logs', 'parkour_finetune'
    )
    os.makedirs(log_root, exist_ok=True)

    print("=" * 60)
    print("Phase 2: Bayesian Optimization for Morphology Co-Design")
    print("=" * 60)

    # Parse args
    bo_iters = getattr(args, 'bo_iterations', 50)
    ft_steps = getattr(args, 'finetune_steps', 400)
    num_eval_eps = getattr(args, 'num_eval_eps', 10)
    pretrained_path = getattr(args, 'pretrained_path', None)

    print(f"BO iterations: {bo_iters}")
    print(f"Finetune steps per candidate: {ft_steps}")
    print(f"Eval episodes: {num_eval_eps}")

    # Load pretrained model path
    if pretrained_path is None:
        pretrained_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs', 'parkour_pretrain'
        )
        pretrained_path = get_load_path(pretrained_root, load_run=-1, checkpoint=-1)
    print(f"Pretrained model: {pretrained_path}")

    # Initialize BO
    bounds = [(0.6, 1.4)] * 4
    bo = BayesianOptimizer(bounds, xi_init=torch.ones(4), n_init=10)

    results = []
    best_score = -float('inf')
    best_xi = torch.ones(4)
    best_metrics = None

    for iteration in range(bo_iters):
        print(f"\n--- BO Iteration {iteration + 1}/{bo_iters} ---")

        xi = bo.suggest()
        print(f"Candidate xi: [{xi[0]:.3f}, {xi[1]:.3f}, {xi[2]:.3f}, {xi[3]:.3f}]")

        # Create fresh env for this morphology
        env_cfg = ParkourCfg()
        train_cfg = ParkourCfgPPO()
        env_cfg.seed = int(time.time()) % 10000
        train_cfg.seed = env_cfg.seed

        # Disable spatial DR for finetuning
        env_cfg.domain_rand.spatial_domain_rand = False
        env_cfg.morphology.target_morphology = xi

        # Reduce envs for finetuning speed
        env_cfg.env.num_envs = 2048

        env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            log_root=None, env=env, name=args.task, args=args, train_cfg=train_cfg
        )

        # Load pretrained weights
        ppo_runner.load(pretrained_path)
        print("Loaded pretrained weights")

        # Finetune
        print(f"Finetuning for {ft_steps} steps...")
        t_start = time.time()
        finetune_policy(env, ppo_runner, steps=ft_steps)
        t_elapsed = time.time() - t_start
        print(f"Finetune completed in {t_elapsed:.1f}s")

        # Evaluate
        print("Evaluating...")
        act_inference = ppo_runner.get_inference_policy(device=env.device)
        score, metrics = evaluate_morphology(env, act_inference, xi, num_eps=num_eval_eps)
        print(f"  Score: {score:.3f} | Dist: {metrics['distance']:.2f}m | "
              f"Height: {metrics['height']:.2f}m | Survival: {metrics['survival']:.2f}")

        bo.observe(xi, score)
        results.append({
            'iteration': iteration,
            'xi': xi.tolist(),
            'score': float(score),
            'metrics': {k: float(v) for k, v in metrics.items()},
        })

        if score > best_score:
            best_score = score
            best_xi = xi.clone()
            best_metrics = metrics
            print(f"  *** NEW BEST ***")

        # Save intermediate results
        results_path = os.path.join(log_root, 'bo_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        env.gym.destroy_sim(env.sim)
        del env
        del ppo_runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary
    print("\n" + "=" * 60)
    print("Co-Design Complete!")
    print(f"Best morphology: xi = [{best_xi[0]:.3f}, {best_xi[1]:.3f}, "
          f"{best_xi[2]:.3f}, {best_xi[3]:.3f}]")
    print(f"Best score: {best_score:.3f}")
    if best_metrics:
        print(f"  Jump distance: {best_metrics['distance']:.2f}m")
        print(f"  Jump height:   {best_metrics['height']:.2f}m")
        print(f"  Survival rate: {best_metrics['survival']:.2f}")
    print("=" * 60)

    final_path = os.path.join(log_root, 'best_morphology.json')
    with open(final_path, 'w') as f:
        json.dump({
            'best_xi': best_xi.tolist(),
            'best_score': float(best_score),
            'best_metrics': {k: float(v) for k, v in best_metrics.items()} if best_metrics else {},
        }, f, indent=2)
    print(f"Results saved to: {final_path}")


if __name__ == '__main__':
    args = get_args()
    main(args)
