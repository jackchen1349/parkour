import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ParkourActorCritic
from rsl_rl.env import VecEnv


class OnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ParkourActorCritic = actor_critic_class(
            num_prop=self.env.cfg.env.n_proprio,
            num_scan=self.env.cfg.env.n_scan,
            num_critic_obs=num_critic_obs,
            num_priv_latent=self.env.cfg.env.n_priv_latent,
            num_priv_explicit=self.env.cfg.env.n_priv,
            num_hist=self.env.cfg.env.history_len,
            num_actions=self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg.get("dagger_update_freq", 20)

        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env,
                              [self.env.num_obs], [self.env.num_privileged_obs],
                              [self.env.num_actions])

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, hist_encoding=hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device), critic_obs.to(self.device),
                        rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, \
                mean_priv_reg_loss, priv_reg_coef = self.alg.update()

            mean_hist_latent_loss = 0.
            if hist_encoding:
                mean_hist_latent_loss = self.alg.update_dagger()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.current_learning_iteration = it
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
            ep_infos.clear()

        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f'model_{tot_iter}.pt'))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs
                  / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/estimator', locs['mean_estimator_loss'], locs['it'])
        self.writer.add_scalar('Loss/hist_latent_loss', locs['mean_hist_latent_loss'], locs['it'])
        self.writer.add_scalar('Loss/priv_reg_loss', locs['mean_priv_reg_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])

        mean_rew = statistics.mean(locs['rewbuffer']) if len(locs['rewbuffer']) > 0 else 0.
        mean_len = statistics.mean(locs['lenbuffer']) if len(locs['lenbuffer']) > 0 else 0.
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', mean_rew, locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', mean_len, locs['it'])

        s = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = (f"""{'#' * width}\n"""
                      f"""{s.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (col: {locs['collection_time']:.2f}s, learn: {locs['learn_time']:.2f}s)\n"""
                      f"""{'Value loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                      f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.6f}\n"""
                      f"""{'Priv reg loss:':>{pad}} {locs['mean_priv_reg_loss']:.6f}\n"""
                      f"""{'Hist latent loss:':>{pad}} {locs['mean_hist_latent_loss']:.6f}\n"""
                      f"""{'Priv reg coef:':>{pad}} {locs['priv_reg_coef']:.4f}\n"""
                      f"""{'Mean action std:':>{pad}} {mean_std.item():.2f}\n"""
                      f"""{'Mean reward:':>{pad}} {mean_rew:.2f}\n"""
                      f"""{'Mean ep length:':>{pad}} {mean_len:.2f}\n""")
        if ep_string:
            log_string += f"""{'-' * width}\n"""
            log_string += ep_string
        log_string += f"""{'-' * width}"""
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'hist_encoder_optimizer_state_dict': self.alg.hist_encoder_optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.estimator_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if 'estimator_state_dict' in loaded_dict:
            self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if 'hist_encoder_optimizer_state_dict' in loaded_dict:
                self.alg.hist_encoder_optimizer.load_state_dict(loaded_dict['hist_encoder_optimizer_state_dict'])
            if 'estimator_optimizer_state_dict' in loaded_dict:
                self.alg.estimator_optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
