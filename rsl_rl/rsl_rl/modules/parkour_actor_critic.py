"""RMA-style asymmetric Actor-Critic for parkour co-design.

Matches extreme-parkour architecture with independent encoders:
  scan_encoder(mt) → scan_latent
  priv_encoder(et) → priv_latent (teacher, has access to privileged info)
  history_encoder(ht) → hist_latent (student, uses proprioceptive history)

  Actor([xt, scan_latent, e't, latent]) → actions(12)
    latent = priv_encoder(et)  in non-hist mode (teacher)
    latent = history_encoder(ht) in hist mode (student / DAgger)
  Critic(raw_privileged_obs) → value(1)  (no encoding, direct raw input)

  Estimator(xt) → predicted_e't  (separate: predicts priv_explicit from proprio)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name):
    if act_name == "elu": return nn.ELU()
    elif act_name == "selu": return nn.SELU()
    elif act_name == "relu": return nn.ReLU()
    elif act_name == "crelu": return nn.CReLU()
    elif act_name == "lrelu": return nn.LeakyReLU()
    elif act_name == "tanh": return nn.Tanh()
    elif act_name == "sigmoid": return nn.Sigmoid()
    else: return None


class StateHistoryEncoder(nn.Module):
    """Conv1D encoder for proprioceptive history ht.

    Input: (batch, tsteps, input_size), Output: (batch, output_size).
    Supports 5/10/20/50 frame histories.
    """
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super().__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        channel_size = 10
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size), self.activation_fn)
        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(30, 20, kernel_size=8, stride=4), self.activation_fn,
                nn.Conv1d(20, 10, kernel_size=5, stride=1), self.activation_fn,
                nn.Conv1d(10, 10, kernel_size=5, stride=1), self.activation_fn, nn.Flatten())
            linear_in = 30
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(30, 20, kernel_size=6, stride=2), self.activation_fn,
                nn.Conv1d(20, 10, kernel_size=4, stride=2), self.activation_fn, nn.Flatten())
            linear_in = 20
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(30, 20, kernel_size=4, stride=2), self.activation_fn,
                nn.Conv1d(20, 10, kernel_size=2, stride=1), self.activation_fn, nn.Flatten())
            linear_in = 30
        elif tsteps == 5:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(30, 20, kernel_size=3, stride=1), self.activation_fn,
                nn.Conv1d(20, 10, kernel_size=3, stride=1), self.activation_fn, nn.Flatten())
            linear_in = 10
        else:
            raise ValueError(f"tsteps={tsteps} not supported")
        self.linear_output = nn.Sequential(nn.Linear(linear_in, output_size), self.activation_fn)

    def forward(self, obs):
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1]))
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        return self.linear_output(output)


class ParkourEstimator(nn.Module):
    """Predicts priv_explicit (e't: base_lin_vel) from proprio (xt) for sim2real.

    NOT used as input to the Actor — only for priv_reg_loss alignment.
    Used at deployment time when real priv_explicit is unavailable.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=None, activation="elu"):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        act = get_activation(activation)
        layers = [nn.Linear(input_dim, hidden_dims[0]), act]
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ParkourActorCritic(nn.Module):
    """RMA-style asymmetric Actor-Critic matching extreme-parkour.

    Observation layout (obs_buf, dim 215):
        [0:47]    proprio (xt)
        [47:50]   priv_explicit (e't)
        [50:83]   priv_latent (et)
        [83:215]  height_scan (mt)

    Privileged observation (450 dims):
        [0:215]   same as obs_buf
        [215:450] history stacked (5 * 47 = 235)

    Architecture:
        ScanEncoder(mt[132]) → scan_latent[scan_encoder_dims[-1]]
        PrivEncoder(et[33]) → priv_latent[priv_encoder_dims[-1]]
        HistoryEncoder(ht: 5×47 Conv1D) → hist_latent[priv_encoder_dims[-1]]
        Actor([xt + scan_latent + e't + latent]) → actions(12)
        Critic(raw_privileged_obs) → value(1)  (no encoding)
    """
    is_recurrent = False

    def __init__(self, num_prop, num_scan, num_critic_obs,
                 num_priv_latent, num_priv_explicit,
                 num_hist, num_actions,
                 scan_encoder_dims=(128, 64, 32),
                 actor_hidden_dims=(512, 256, 128),
                 critic_hidden_dims=(512, 256, 128),
                 priv_encoder_dims=(64, 32),
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ParkourActorCritic: ignoring kwargs: " + str(list(kwargs.keys())))
        super().__init__()
        act = get_activation(activation)
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.num_hist = num_hist
        self.num_actions = num_actions

        # ---- Scan encoder: mt → scan_latent ----
        if len(scan_encoder_dims) > 0 and num_scan > 0:
            scan_enc_layers = []
            scan_enc_layers.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_enc_layers.append(act)
            for l in range(len(scan_encoder_dims)):
                if l == len(scan_encoder_dims) - 1:
                    scan_enc_layers.append(nn.Linear(scan_encoder_dims[l-1], scan_encoder_dims[l]))
                    scan_enc_layers.append(nn.Tanh())
                elif l > 0:
                    scan_enc_layers.append(nn.Linear(scan_encoder_dims[l-1], scan_encoder_dims[l]))
                    scan_enc_layers.append(act)
            self.scan_encoder = nn.Sequential(*scan_enc_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        # ---- Privileged encoder: et → priv_latent (teacher) ----
        if len(priv_encoder_dims) > 0:
            priv_enc_layers = []
            priv_enc_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
            priv_enc_layers.append(act)
            for l in range(len(priv_encoder_dims) - 1):
                priv_enc_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                priv_enc_layers.append(act)
            self.priv_encoder = nn.Sequential(*priv_enc_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        # ---- History encoder: ht → hist_latent (student) ----
        self.history_encoder = StateHistoryEncoder(
            act, num_prop, num_hist, priv_encoder_output_dim)

        # ---- Actor: [xt + scan_latent + e't + latent] → actions ----
        actor_in = (num_prop + self.scan_encoder_output_dim
                    + num_priv_explicit + priv_encoder_output_dim)
        actor_layers = [nn.Linear(actor_in, actor_hidden_dims[0]), act]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(act)
        self.actor = nn.Sequential(*actor_layers)

        # ---- Critic: raw privileged obs → value (NO encoding) ----
        critic_in = (num_priv_latent + num_scan + num_priv_explicit
                     + num_prop + num_hist * num_prop)
        critic_layers = [nn.Linear(critic_in, critic_hidden_dims[0]), act]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(act)
        self.critic = nn.Sequential(*critic_layers)

        print(f"ParkourActorCritic: "
              f"ScanEncoder {num_scan}→{self.scan_encoder_output_dim}, "
              f"PrivEncoder {num_priv_latent}→{priv_encoder_output_dim}, "
              f"HistoryEncoder({num_hist}fx{num_prop})→{priv_encoder_output_dim}, "
              f"Actor {actor_in}→{num_actions}, Critic {critic_in}→1")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None): pass
    def forward(self): raise NotImplementedError

    @property
    def action_mean(self): return self.distribution.mean
    @property
    def action_std(self): return self.distribution.stddev
    @property
    def entropy(self): return self.distribution.entropy().sum(dim=-1)

    def _slice_obs(self, obs):
        """Slice obs_buf into components."""
        proprio = obs[:, :self.num_prop]
        priv_explicit = obs[:, self.num_prop:self.num_prop + self.num_priv_explicit]
        priv_latent = obs[:, self.num_prop + self.num_priv_explicit:
                          self.num_prop + self.num_priv_explicit + self.num_priv_latent]
        height_scan = obs[:, self.num_prop + self.num_priv_explicit + self.num_priv_latent:
                          self.num_prop + self.num_priv_explicit + self.num_priv_latent + self.num_scan]
        return proprio, priv_explicit, priv_latent, height_scan

    def _slice_privileged_obs(self, privileged_obs):
        """Slice privileged observation: obs_buf(215) + history(5*47=235)."""
        obs_len = self.num_prop + self.num_priv_explicit + self.num_priv_latent + self.num_scan
        obs_buf = privileged_obs[:, :obs_len]
        history = privileged_obs[:, -self.num_hist * self.num_prop:]
        return obs_buf, history

    def infer_scan_latent(self, obs):
        _, _, _, height_scan = self._slice_obs(obs)
        return self.scan_encoder(height_scan)

    def infer_priv_latent(self, obs):
        _, _, priv_latent, _ = self._slice_obs(obs)
        return self.priv_encoder(priv_latent)

    def infer_hist_latent(self, privileged_obs):
        _, history = self._slice_privileged_obs(privileged_obs)
        return self.history_encoder(history.view(-1, self.num_hist, self.num_prop))

    def _build_actor_input(self, observations, hist_encoding=False):
        proprio, priv_explicit, _, height_scan = self._slice_obs(observations)
        scan_latent = self.scan_encoder(height_scan)
        if hist_encoding:
            latent = self.infer_hist_latent(observations)
        else:
            latent = self.infer_priv_latent(observations)
        return torch.cat([proprio, scan_latent, priv_explicit, latent], dim=-1)

    def update_distribution(self, observations, hist_encoding=False):
        actor_input = self._build_actor_input(observations, hist_encoding)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, **kwargs):
        """Deterministic inference for deployment."""
        actor_input = self._build_actor_input(observations, hist_encoding)
        return self.actor(actor_input)

    def evaluate(self, critic_observations, **kwargs):
        """Critic takes raw privileged observation directly — NO encoding."""
        return self.critic(critic_observations)
