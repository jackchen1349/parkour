"""ParkourRobot — goal-driven parkour and morphology co-design.

Extends LeggedRobot with:
- Goal/waypoint tracking (ref extreme-parkour _update_goals)
- Multi-morphology support via MorphologyManager
- PD gain correction for morphology scaling
- Privileged observations with morphology parameters
- Terrain curriculum with goal updates
"""

import os
import tempfile
import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.parkour.morphology import MorphologyManager
from legged_gym.envs.parkour.parkour_terrain import ParkourHeightField
from legged_gym.envs.parkour.parkour_rewards import ParkourRewards


def euler_from_quaternion(quat_angle):
    x = quat_angle[:, 0]; y = quat_angle[:, 1]; z = quat_angle[:, 2]; w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1., 1.)
    pitch = torch.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    return roll, pitch, yaw


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
        self.global_counter = 0
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ---- Asset loading (multi-morphology) ----

    def _load_asset_from_urdf(self, urdf_str, prefix):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.urdf', delete=False,
            dir=os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/robots/parkour_quadruped/urdf'),
            prefix=prefix
        ) as f:
            f.write(urdf_str)
            tmp_path = f.name
        asset = self.gym.load_asset(self.sim, os.path.dirname(tmp_path),
                                    os.path.basename(tmp_path), self._create_asset_options())
        os.unlink(tmp_path)
        return asset

    def _load_robot_asset(self):
        if not self._spatial_dr:
            xi = getattr(self.cfg.morphology, 'target_morphology', None)
            if xi is None:
                xi = torch.ones(4)
            else:
                xi = xi.float()
            self._morphology_params_per_env = xi.unsqueeze(0).repeat(self.num_envs, 1)
            urdf_str = self._morphology_manager.build_urdf_string(xi)
            asset = self._load_asset_from_urdf(urdf_str, 'finetune_')
            self._assets = [asset]
            return asset

        num_buckets = self.cfg.morphology.num_buckets
        xi_samples = self._morphology_manager.sample_morphologies(num_buckets)
        self._morphology_params = xi_samples

        assets = [
            self._load_asset_from_urdf(
                self._morphology_manager.build_urdf_string(xi_samples[i]),
                f'morph_{i:03d}_'
            )
            for i in range(num_buckets)
        ]

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

        if mesh_type in ['heightfield', 'trimesh']:
            self.height_samples = torch.tensor(self.terrain.heightsamples).view(
                self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

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
                self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3,
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
        self.env_goals[:] = torch.cat(
            (temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
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

        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(
            self.terrain.tot_rows, self.terrain.tot_cols
        ).to(self.device)

    # ---- Observations (matches extreme-parkour: all info in single obs_buf) ----

    def compute_observations(self):
        """obs_buf = [xt(47) | mt(132) | e't(3) | et(33) | ht(5*47=235)] = 450
           No separate privileged_obs_buf (num_privileged_obs=None).
        """
        # Update yaw deltas every 5 steps (ref extreme-parkour L388-390)
        if self.global_counter % 5 == 0:
            self.delta_yaw = self.target_yaw - self.yaw
            self.delta_next_yaw = self.next_target_yaw - self.yaw

        # xt: proprioceptive state [47]
        xt = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,                               # 3
            self.roll[:, None],                                                         # 1
            self.pitch[:, None],                                                        # 1
            self.delta_yaw[:, None],                                                    # 1
            self.delta_next_yaw[:, None],                                               # 1
            (self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos,        # 12
            self.dof_vel * self.obs_scales.dof_vel,                                     # 12
            self.actions,                                                                # 12
            self.contact_filt.float() - 0.5,                                            # 4
        ), dim=-1)  # = 47

        # e't: explicit privileged — body linear velocity [3]
        et_prime = self.base_lin_vel * self.obs_scales.lin_vel

        # et: privileged latent [33]
        et = torch.cat((
            self._body_total_mass[:, None],                                              # 1
            self._body_com,                                                              # 3
            self.mass_params_tensor,                                                     # 4
            self.friction_coeffs_tensor,                                                 # 1
            self.motor_strength[0] - 1.,                                                 # 12
            self.motor_strength[1] - 1.,                                                 # 12
        ), dim=-1)  # = 33

        # mt: exteroceptive — height samples [132]
        heights = self.root_states[:, 2].unsqueeze(1) - 0.35 - self.measured_heights
        mt = torch.clip(heights, -1., 1.) * self.obs_scales.height_measurements

        # obs_buf = [xt | mt | e't | et | ht_flattened] = 47+132+3+33+235 = 450
        self.obs_buf = torch.cat([
            xt,
            mt,
            et_prime,
            et,
            self.obs_history_buf.view(self.num_envs, -1),
        ], dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # ---- update history: store xt (47 dims) with yaw masked (ref extreme-parkour L420) ----
        xt_masked = self.obs_buf[:, :self.cfg.env.n_proprio].clone()
        xt_masked[:, 5:7] = 0.  # mask delta_yaw(5), delta_next_yaw(6)
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            xt_masked.unsqueeze(1).repeat(1, self.cfg.env.history_len, 1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                xt_masked.unsqueeze(1)
            ], dim=1),
        )

    def _get_noise_scale_vec(self, cfg):
        n_prop = cfg.env.n_proprio
        n_priv = cfg.env.n_priv
        n_priv_latent = cfg.env.n_priv_latent
        n_scan = cfg.env.n_scan

        noise_vec = torch.zeros(self.num_obs, device=self.device)
        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level
        # xt: ang_vel(3)
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # xt: roll(1), pitch(1), delta_yaw(1), delta_next_yaw(1)
        noise_vec[3:7] = noise_scales.gravity * noise_level
        # xt: dof_pos(12)
        noise_vec[7:19] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # xt: dof_vel(12)
        noise_vec[19:31] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # mt: height scan [47:179] (after xt in new layout)
        mt_start = n_prop  # xt(47)
        noise_vec[mt_start:mt_start + n_scan] = (
            noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements)
        return noise_vec

    def _compute_torques(self, actions):
        """Paper Eq.9-11 torque chain:
           τ_target = k_p(q* − q) + k_d(q̇* − q̇)    (Eq.9)
           τ_scale  = τ_target × α                     (Eq.10, motor strength)
           τ_real   = clip(τ_scale, −τ_max, τ_max)    (Eq.11)
           PD gains k_i = η_i × k̃_i per Eq.2, where η_i = aξ³+bξ²+cξ+d (Eq.1)
           Base PD gains: k̃_p=40, k̃_d=0.7 (Lite3).
        """
        actions_scaled = actions * self.cfg.control.action_scale

        if self._spatial_dr and self._morphology_params_per_env is not None:
            corrections = self._morphology_manager.compute_pd_corrections(
                self._morphology_params_per_env
            ).to(self.device)
            p_gains = self.p_gains.unsqueeze(0) * corrections
            d_gains = self.d_gains.unsqueeze(0) * corrections
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        # τ_target = k_p * Δq + k_d * (−q̇)   (desired velocity q̇* = 0)
        torques = (self.motor_strength[0] * p_gains           # α × k_p
                   * (actions_scaled + self.default_dof_pos_all - self.dof_pos)
                   - self.motor_strength[1] * d_gains         # α × k_d
                   * self.dof_vel)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # ---- Goal tracking (ref extreme-parkour _update_goals) ----

    @staticmethod
    def _pos_to_yaw(pos_rel):
        norm = torch.norm(pos_rel, dim=-1, keepdim=True)
        vec = pos_rel / (norm + 1e-5)
        return torch.atan2(vec[:, 1], vec[:, 0])

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

        self.target_yaw = self._pos_to_yaw(self.target_pos_rel)
        self.next_target_yaw = self._pos_to_yaw(self.next_target_pos_rel)

    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.feet_state = self.rigid_body_states.view(self.num_envs, -1, 13)[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.last_torques = torch.zeros_like(self.torques)

        hip_names = [s for s in self.dof_names if "Hip" in s]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)

        self.default_dof_pos_all = self.default_dof_pos.repeat(self.num_envs, 1)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device)
        self.roll = torch.zeros(self.num_envs, device=self.device)
        self.pitch = torch.zeros(self.num_envs, device=self.device)
        self.yaw = torch.zeros(self.num_envs, device=self.device)
        self._body_total_mass = torch.ones(self.num_envs, device=self.device)
        self._body_com = torch.zeros(self.num_envs, 3, device=self.device)
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = torch.zeros(self.num_envs, self.cfg.env.n_scan, device=self.device)

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        if self._morphology_params_per_env is not None:
            self.mass_params_tensor[:] = self._morphology_params_per_env.to(self.device)
        self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).float().view(self.num_envs, 1)
        if hasattr(self, '_body_masses'):
            self._body_total_mass = self._body_masses.sum(dim=1)

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        if env_id == 0:
            self._body_masses = torch.zeros(self.num_envs, self.num_bodies, device=self.device)
        for b in range(len(props)):
            self._body_masses[env_id, b] = props[b].mass
        return props

    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.8 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            self.commands[:, 2] *= torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.global_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        if hasattr(self, '_body_masses'):
            weighted_pos = self.rigid_body_states[:, :, :3] * self._body_masses[:, :, None]
            self._body_com[:] = weighted_pos.sum(dim=1) / self._body_total_mass[:, None]

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        self._update_goals()
        self._post_physics_step_callback()

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    # ---- Reset and curriculum ----

    def _reset_root_states(self, env_ids):
        n = len(env_ids)
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (n, 2), device=self.device)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range * torch_rand_float(-1, 1, (n, 1), device=self.device).squeeze(1)
                quat = quat_from_euler_xyz(torch.zeros(n, device=self.device), torch.zeros(n, device=self.device), rand_yaw)
                self.root_states[env_ids, 3:7] = quat[:]
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (n, 1), device=self.device).squeeze(1)
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
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0
        self.delta_yaw[env_ids] = 0.
        self.delta_next_yaw[env_ids] = 0.

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(
                self.episode_sums[key][env_ids]
            ) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s
        move_up = distance > 0.8 * threshold
        move_down = distance < 0.4 * threshold
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
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        self.reset_buf |= torch.abs(self.roll) > 1.5
        self.reset_buf |= torch.abs(self.pitch) > 1.5
        self.reset_buf |= self.root_states[:, 2] < -0.25

        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.time_out_buf |= reach_goal_cutoff

        self.reset_buf |= self.time_out_buf

    # ---- Height Measurements ----

    def _init_height_points(self):
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points, 2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points, 2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def _get_heights(self, env_ids=None):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
