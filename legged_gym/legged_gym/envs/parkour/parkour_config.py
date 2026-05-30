"""Parkour co-design config — terrain/reward params reference extreme-parkour.

Terrain proportions: 19 types (indices 0-19), proportions sum to ~2.0 and normalized.
Parkour-specific terrains (idx 15-19): parkour, parkour_hurdle, parkour_flat,
  parkour_step (high jump), parkour_gap (long jump) — each 0.2 proportion.

Reward names match extreme-parkour: tracking_goal_vel, tracking_yaw, delta_torques, etc.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class ParkourCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 6144  # paper: N = 8192 parallel environments
        n_proprio = 46  # 3(ang_vel)+1(roll)+1(pitch)+1(delta_yaw)+12(dof_pos)+12(dof_vel)+12(prev_action)+4(contact)
        history_len = 5
        num_observations = 46
        # privileged = x_t(46) + lin_vel(3) + mass(1) + COM(3) + morph(4) + friction(1) + motor_kp(12) + motor_kd(12) + history(230) = 312
        num_privileged_obs = 312
        num_actions = 12
        episode_length_s = 20
        send_timeouts = True
        env_spacing = 3.
        test = False

        next_goal_threshold = 0.2
        reach_goal_delay = 0.1

        randomize_start_pos = True
        randomize_start_yaw = True
        rand_yaw_range = 1.2
        randomize_start_y = True
        rand_y_range = 0.8

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        horizontal_scale = 0.05
        vertical_scale = 0.005
        edge_width_thresh = 0.05
        border_size = 5
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = False
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        y_range = [-0.4, 0.4]
        height = [0.02, 0.06]
        downsampled_scale = 0.075
        measure_horizontal_noise = 0.0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        slope_treshold = 1.5
        origin_zero_z = True
        max_init_terrain_level = 5
        num_goals = 8
        edge_width_thresh = 0.05
        simplify_grid = False

        # Paper focuses on two parkour tasks: long jump (gap) and high jump (step).
        # Pretraining uses diverse terrains; finetuning evaluates on gap/step only.
        # 20 terrain type proportions (normalized to sum=1.0):
        terrain_proportions = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.10, 0.30, 0.30, 0.0,
        ]

    class morphology:
        scaling_range = [0.6, 1.4]
        num_buckets = 100
        pd_correction_coeffs = [0.0, 0.0, 1.0, 0.0]
        target_morphology = None

    class parkour:
        gap_width = 1.0
        platform_height = 0.55
        waypoint_update_freq = 0.5
        forward_speed_target = 2.0

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 10.
        heading_command = False
        curriculum = False
        lin_vel_clip = 0.2
        ang_vel_clip = 0.4

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.375]
        default_joint_angles = {
            'FL_HipX_joint': 0.0, 'FL_HipY_joint': -0.65, 'FL_Knee_joint': 1.3,
            'FR_HipX_joint': 0.0, 'FR_HipY_joint': -0.65, 'FR_Knee_joint': 1.3,
            'HL_HipX_joint': 0.0, 'HL_HipY_joint': -0.65, 'HL_Knee_joint': 1.3,
            'HR_HipX_joint': 0.0, 'HR_HipY_joint': -0.65, 'HR_Knee_joint': 1.3,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'HipX': 30.0, 'HipY': 30.0, 'Knee': 30.0}
        damping = {'HipX': 1.0, 'HipY': 1.0, 'Knee': 1.0}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/parkour_quadruped/urdf/parkour_quadruped.urdf'
        name = "parkour_quadruped"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK", "HIP"]
        terminate_after_contacts_on = ["TORSO"]
        self_collisions = 0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]
        push_robots = True
        push_interval_s = 15.0
        max_push_vel_xy = 1.0
        spatial_domain_rand = True
        motor_strength_range = [0.8, 1.2]
        action_buf_len = 8

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.35
        max_contact_force = 100.0

        class scales:
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_acc = -2.5e-7
            collision = -10.0
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.00001
            hip_pos = -0.5
            dof_error = -0.04
            feet_stumble = -1.0
            feet_edge = -1.0
            dof_pos_limits = 0.0
            termination = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            stand_still = 0.0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 1.2

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class ParkourCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = 'adaptive'
        gamma = 0.98
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 6000
        save_interval = 100
        experiment_name = 'parkour_pretrain'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
