"""Parkour reward functions — reference extreme-parkour implementations.

Reward terms:
  tracking_goal_vel (1.5)  tracking_yaw (0.5)    lin_vel_z (-1.0)
  ang_vel_xy (-0.05)       orientation (-1.0)     dof_acc (-2.5e-7)
  collision (-10.)         action_rate (-0.1)     delta_torques (-1.0e-7)
  torques (-0.00001)       hip_pos (-0.5)         dof_error (-0.04)
  feet_stumble (-1)        feet_edge (-1)

Formulas match extreme-parkour legged_robot.py lines 1224-1287.
"""

import torch


class ParkourRewards:

    def _reward_tracking_goal_vel(self):
        """Vel toward goal, capped at command. Ref: extreme-parkour L1226."""
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1),
                           self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        return rew

    def _reward_tracking_yaw(self):
        """Exp penalty on yaw error. Ref: extreme-parkour L1233."""
        return torch.exp(-torch.abs(self.target_yaw - self.yaw))

    def _reward_lin_vel_z(self):
        """Penalize z velocity, scaled by terrain type. Ref: extreme-parkour L1237."""
        rew = torch.square(self.base_lin_vel[:, 2])
        if hasattr(self, 'env_class'):
            rew[self.env_class != 17] *= 0.5
        return rew

    def _reward_ang_vel_xy(self):
        """Penalize xy angular velocity. Ref: extreme-parkour L1242."""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """Penalize non-flat orientation (on specific terrains). Ref: extreme-parkour L1245."""
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        if hasattr(self, 'env_class'):
            rew[self.env_class != 17] = 0.
        return rew

    def _reward_dof_acc(self):
        """Penalize joint acceleration. Ref: extreme-parkour L1250."""
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        """Penalize body collisions. Ref: extreme-parkour L1253."""
        return torch.sum(
            1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1
        )

    def _reward_action_rate(self):
        """Penalize action changes. Ref: extreme-parkour L1256."""
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        """Penalize torque changes. Ref: extreme-parkour L1259."""
        if self.last_torques is None or self.last_torques.shape != self.torques.shape:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_torques(self):
        """Penalize joint torques. Ref: extreme-parkour L1262."""
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_hip_pos(self):
        """Penalize hip joint deviation. Ref: extreme-parkour L1265."""
        if not hasattr(self, 'hip_indices'):
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(
            torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]),
            dim=1
        )

    def _reward_dof_error(self):
        """Penalize all joint deviation from default. Ref: extreme-parkour L1268."""
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error

    def _reward_feet_stumble(self):
        """Penalize horizontal foot contact forces. Ref: extreme-parkour L1272."""
        rew = torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
            4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1
        )
        return rew.float()

    def _reward_feet_edge(self):
        """Penalize feet on terrain edges (high difficulty only). Ref: extreme-parkour L1278."""
        if not hasattr(self, 'x_edge_mask') or not hasattr(self, 'contact_filt'):
            return torch.zeros(self.num_envs, device=self.device)
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] +
                       self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew
