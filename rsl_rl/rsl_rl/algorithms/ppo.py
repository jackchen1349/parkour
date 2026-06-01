"""PPO with RMA-style asymmetric training for parkour co-design.

Matches extreme-parkour architecture:
  - scan_encoder(mt) → scan_latent
  - priv_encoder(et) → priv_latent (teacher, sees privileged info)
  - history_encoder(ht) → hist_latent (student, uses proprioceptive history)
  - Actor([xt, scan_latent, e't, latent]) → actions
  - Critic(raw_privileged_obs) → value (no encoding)

  - Estimator(xt) → predicted_e't (separate sim2real module)
  - priv_reg_loss: ||priv_encoder(et) - history_encoder(ht)||² (DAgger alignment)

train_with_estimated_states: replace e't with Estimator(xt) during act()
priv_reg_coef_schedual: [start_coef, end_coef, start_iter, duration]
"""

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ParkourActorCritic
from rsl_rl.storage import RolloutStorage


class PPO:
    actor_critic: ParkourActorCritic

    def __init__(self, actor_critic, num_learning_epochs=1, num_mini_batches=1,
                 clip_param=0.2, gamma=0.998, lam=0.95, value_loss_coef=1.0,
                 entropy_coef=0.0, learning_rate=1e-3, max_grad_norm=1.0,
                 use_clipped_value_loss=True, schedule="fixed", desired_kl=0.01,
                 device='cpu', dagger_update_freq=20,
                 priv_reg_coef_schedual=None,
                 estimator_hidden_dims=None,
                 train_with_estimated_states=True,
                 **kwargs):
        if priv_reg_coef_schedual is None:
            priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        if estimator_hidden_dims is None:
            estimator_hidden_dims = [256, 128]
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None

        # Separate optimizers for each encoder module
        actor_critic_params = set(self.actor_critic.actor.parameters())
        critic_params = set(self.actor_critic.critic.parameters())
        scan_encoder_params = set(self.actor_critic.scan_encoder.parameters())
        priv_encoder_params = set(self.actor_critic.priv_encoder.parameters())
        hist_encoder_params = set(self.actor_critic.history_encoder.parameters())
        # std is a single parameter
        base_params = [self.actor_critic.std]
        # Actor + Critic + scan_encoder + priv_encoder optimizer
        ppo_params = (list(actor_critic_params | critic_params | scan_encoder_params | priv_encoder_params)
                      + base_params)
        self.optimizer = optim.Adam(ppo_params, lr=learning_rate)
        # History encoder optimizer (for DAgger)
        self.hist_encoder_optimizer = optim.Adam(hist_encoder_params, lr=learning_rate)

        # Estimator: predicts priv_explicit (e't) from proprio (xt) for sim2real
        from rsl_rl.modules.parkour_actor_critic import ParkourEstimator
        self.estimator = ParkourEstimator(
            input_dim=actor_critic.num_prop,
            output_dim=actor_critic.num_priv_explicit,
            hidden_dims=estimator_hidden_dims,
            activation='elu',
        ).to(self.device)
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=learning_rate)
        self.train_with_estimated_states = train_with_estimated_states

        self.transition = RolloutStorage.Transition()
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.dagger_update_freq = dagger_update_freq
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env,
                                      actor_obs_shape, critic_obs_shape,
                                      action_shape, self.device)

    def test_mode(self): self.actor_critic.test()
    def train_mode(self): self.actor_critic.train()

    def act(self, obs, critic_obs, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Optionally replace priv_explicit (e't) with Estimator prediction for sim2real
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            proprio = obs_est[:, :self.actor_critic.num_prop]
            priv_explicit_estimated = self.estimator(proprio)
            obs_est[:, self.actor_critic.num_prop:
                    self.actor_critic.num_prop + self.actor_critic.num_priv_explicit] = priv_explicit_estimated
            self.transition.actions = self.actor_critic.act(
                obs_est, hist_encoding, privileged_obs=critic_obs).detach()
        else:
            self.transition.actions = self.actor_critic.act(
                obs, hist_encoding, privileged_obs=critic_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        mean_priv_reg_loss = 0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, \
            advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            self.actor_critic.act(obs_batch, hist_encoding=False)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # ---- priv_reg_loss: align priv_encoder(et) with history_encoder(ht) ----
            priv_latent = self.actor_critic.infer_priv_latent(obs_batch)
            with torch.inference_mode():
                hist_latent = self.actor_critic.infer_hist_latent(critic_obs_batch)
            priv_reg_loss = (priv_latent - hist_latent.detach()).norm(p=2, dim=1).mean()
            priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0)
                                 / self.priv_reg_coef_schedual[3], 1)
            priv_reg_coef = (priv_reg_stage * (self.priv_reg_coef_schedual[1]
                             - self.priv_reg_coef_schedual[0])
                             + self.priv_reg_coef_schedual[0])

            # ---- Estimator: predict priv_explicit (e't) from proprio (xt) ----
            _, true_priv_explicit, _, _ = self.actor_critic._slice_obs(obs_batch)
            priv_explicit_pred = self.estimator(obs_batch[:, :self.actor_critic.num_prop])
            estimator_loss = (priv_explicit_pred - true_priv_explicit).pow(2).mean()
            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()

            # ---- Adaptive learning rate ----
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                                   + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                                   / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # ---- PPO surrogate loss ----
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ---- Value loss ----
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param)
                value_loss = torch.max((value_batch - returns_batch).pow(2),
                                       (value_clipped - returns_batch).pow(2)).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (surrogate_loss + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + priv_reg_coef * priv_reg_loss)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimator_loss += estimator_loss.item()
            mean_priv_reg_loss += priv_reg_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self.counter += 1
        return (mean_value_loss / num_updates, mean_surrogate_loss / num_updates,
                mean_estimator_loss / num_updates, mean_priv_reg_loss / num_updates,
                priv_reg_coef)

    def update_dagger(self):
        """DAgger: train history_encoder to match priv_encoder output."""
        mean_hist_latent_loss = 0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, \
            advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            with torch.inference_mode():
                priv_latent = self.actor_critic.infer_priv_latent(obs_batch)
            hist_latent = self.actor_critic.infer_hist_latent(critic_obs_batch)
            hist_latent_loss = (priv_latent.detach() - hist_latent).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.history_encoder.parameters(),
                                    self.max_grad_norm)
            self.hist_encoder_optimizer.step()
            mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self.counter += 1
        return mean_hist_latent_loss / num_updates
