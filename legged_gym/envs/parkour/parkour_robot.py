"""ParkourRobot — goal-driven parkour and morphology co-design.

Extends LeggedRobot with:
- Goal/waypoint tracking (ref extreme-parkour _update_goals)
- Multi-morphology support via MorphologyManager
- PD gain correction for morphology scaling
- Privileged observations with morphology parameters
- Terrain curriculum with goal updates
"""

import os
import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.parkour.morphology import MorphologyManager
from legged_gym.envs.parkour.parkour_terrain import ParkourHeightField
from legged_gym.envs.parkour.parkour_rewards import ParkourRewards


class ParkourRobot(LeggedRobot, ParkourRewards):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._morphology_manager = MorphologyManager(
            scaling_range=cfg.morphology.scaling_range,
            pd_coeffs=cfg.morphology.pd_correction_coeffs,
        )
        self._spatial_dr = cfg.domain_rand.spatial_domain_rand
        self._morphology_params_per_env = None
        self._assets = []
        self._asset_indices = None
        self.last_torques = None
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ---- Asset loading (multi-morphology) ----

    def _load_robot_asset(self):
        if not self._spatial_dr:
            xi = getattr(self.cfg.morphology, 'target_morphology', None)
            if xi is None:
                xi = torch.ones(4)
            else:
                xi = xi.float()
            self._morphology_params_per_env = xi.unsqueeze(0).repeat(self.num_envs, 1)
            urdf_str = self._morphology_manager.build_urdf_string(xi)
            asset_options = self._create_asset_options()
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.urdf', delete=False,
                dir=os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/robots/parkour_quadruped/urdf'),
                prefix='finetune_'
            ) as f:
                f.write(urdf_str)
                tmp_path = f.name
            asset = self.gym.load_asset(self.sim, os.path.dirname(tmp_path), os.path.basename(tmp_path), asset_options)
            self._assets = [asset]
            os.unlink(tmp_path)
            return asset

        num_buckets = self.cfg.morphology.num_buckets
        xi_samples = self._morphology_manager.sample_morphologies(num_buckets)
        self._morphology_params = xi_samples

        assets = []
        asset_options = self._create_asset_options()
        for i in range(num_buckets):
            urdf_str = self._morphology_manager.build_urdf_string(xi_samples[i])
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.urdf', delete=False,
                dir=os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/robots/parkour_quadruped/urdf'),
                prefix=f'morph_{i:03d}_'
            ) as f:
                f.write(urdf_str)
                tmp_path = f.name
            asset = self.gym.load_asset(self.sim, os.path.dirname(tmp_path), os.path.basename(tmp_path), asset_options)
            assets.append(asset)
            os.unlink(tmp_path)

        self._assets = assets
        env_ids = torch.arange(self.num_envs)
        bucket_ids = env_ids % num_buckets
        self._asset_indices = bucket_ids
        self._morphology_params_per_env = xi_samples[bucket_ids]
        return assets[0]

    def _create_asset_options(self):
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        return asset_options

    # ---- Simulation ----

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = ParkourHeightField(self.cfg, self.num_envs)

        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(f"Terrain mesh type not recognised: {mesh_type}")

        self._create_envs()

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

            max_init_level = min(self.cfg.terrain.max_init_terrain_level, self.cfg.terrain.num_rows - 1)
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1

            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode='floor'
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

            # Goal system
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(
                self.num_envs, self.cfg.terrain.num_goals + 2, 3,
                device=self.device, requires_grad=False
            )
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._update_env_goals()

            # x_edge_mask for feet_edge reward
            if hasattr(self.terrain, 'x_edge_mask'):
                self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(
                    self.terrain.tot_rows, self.terrain.tot_cols
                ).to(self.device)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _update_env_goals(self):
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, 2, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(
            1, (self.cur_goal_idx[:, None, None] + future).expand(-1, -1, self.env_goals.shape[-1])
        ).squeeze(1)

    # ---- Heightfield / trimesh ----

    def _create_heightfield(self):
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order='C'), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows, self.terrain.tot_cols
        ).to(self.device)

    def _create_trimesh(self):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order='C'),
            self.terrain.triangles.flatten(order='C'),
            tm_params
        )
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows, self.terrain.tot_cols
        ).to(self.device)

    # ---- Observations ----

    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

        if self.num_privileged_obs is not None:
            if self._morphology_params_per_env is not None:
                morph_params = self._morphology_params_per_env.to(self.device)
            else:
                morph_params = torch.ones(self.num_envs, 4, device=self.device)
            self.privileged_obs_buf = torch.cat([
                self.obs_buf,
                self.base_lin_vel * self.obs_scales.lin_vel,
                morph_params,
            ], dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # ---- PD Control with morphology correction ----

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale

        if self._spatial_dr and self._morphology_params_per_env is not None:
            p_gains = self.p_gains.clone().unsqueeze(0).repeat(self.num_envs, 1)
            d_gains = self.d_gains.clone().unsqueeze(0).repeat(self.num_envs, 1)
            for i in range(min(self.num_envs, self._morphology_params_per_env.shape[0])):
                correction = self._morphology_manager.compute_pd_corrections(
                    self._morphology_params_per_env[i]
                ).to(self.device)
                p_gains[i] *= correction
                d_gains[i] *= correction
            torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel
        else:
            torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # ---- Goal tracking (ref extreme-parkour _update_goals) ----

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = torch.norm(
            self.root_states[:, :2] - self.cur_goals[:, :2], dim=1
        ) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        self._update_goals()

    def post_physics_step(self):
        super().post_physics_step()
        if self.last_torques is None:
            self.last_torques = torch.zeros_like(self.torques)
        self.last_torques[:] = self.torques[:]

    # ---- Reset and curriculum ----

    def _reset_root_states(self, env_ids):
        n = len(env_ids)
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 0] -= 1.2 + torch_rand_float(-0.3, 0.3, (n, 1), device=self.device).squeeze(1)
            self.root_states[env_ids, 1] += torch_rand_float(-0.8, 0.8, (n, 1), device=self.device).squeeze(1)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.2, 0.2, (n, 6), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self._update_terrain_curriculum(env_ids)
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(
                self.episode_sums[key][env_ids]
            ) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.terrain.env_length / 2
        move_down = (distance < 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)
        )
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self._update_env_goals()

    # ---- Termination ----

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
            dim=1
        )
        self.reset_buf |= torch.logical_or(
            torch.abs(self.rpy[:, 1]) > 1.0,
            torch.abs(self.rpy[:, 0]) > 0.8
        )
        self.reset_buf |= self.root_states[:, 2] < 0.1

        # Also terminate if all goals reached
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.time_out_buf |= reach_goal_cutoff

        self.reset_buf |= self.time_out_buf

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = self.cfg.parkour.forward_speed_target
        self.commands[env_ids, 1] = 0.
        self.commands[env_ids, 2] = 0.
        self.commands[env_ids, 3] = 0.

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self._waypoint_resample_interval = int(
            cfg.parkour.waypoint_update_freq / (self.dt + 1e-6)
        )
