"""Parkour co-design config — RMA-style architecture matching extreme-parkour.

Observation layout (per paper Sec 2.1):
  xt  (n_proprio=47): ang_vel(3)+roll(1)+pitch(1)+delta_yaw(1)+delta_next_yaw(1)
                       +dof_pos(12)+dof_vel(12)+prev_action(12)+contact(4)
  e't (n_priv=3):      base_lin_vel (explicit privileged)
  et  (n_priv_latent=33): mass(1)+com(3)+morph(4)+friction(1)+motor_kp(12)+motor_kd(12)
  mt  (n_scan=132):      height measurements 12x11 grid (exteroceptive)
  ht  (history=5x47=235): stacked proprioceptive history

  obs_buf = [xt, e't, et, mt] = 215
  privileged_obs_buf = [xt, e't, et, mt, ht] = 450

Network (matching extreme-parkour ActorCriticRMA):
  ScanEncoder(mt) → scan_latent
  PrivEncoder(et) → priv_latent (teacher)
  HistoryEncoder(ht) → hist_latent (student / DAgger)
  Actor([xt, scan_latent, e't, latent]) → actions(12)
  Critic(raw_privileged_obs) → value(1)  ← no encoding, direct raw input
  Estimator(xt) → predicted_e't  ← separate sim2real module
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class ParkourCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 6144
        n_proprio = 47
        n_scan = 132
        n_priv = 3
        n_priv_latent = 33
        history_len = 5
        history_encoding = True
        num_observations = 450  # n_proprio(47)+n_scan(132)+n_priv(3)+n_priv_latent(33)+history_len*n_proprio(235)
        num_privileged_obs = None  # all info in obs_buf, matching extreme-parkour
        num_actions = 12
        episode_length_s = 20
        send_timeouts = True
        env_spacing = 3.
        test = False

        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

        randomize_start_pos = True
        randomize_start_yaw = True
        rand_yaw_range = 1.2
        randomize_start_y = True
        rand_y_range = 0.8

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        horizontal_scale = 0.05
        vertical_scale = 0.005
        border_size = 5
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
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
        hf2mesh_method = 'grid'
        edge_width_thresh = 0.05
        simplify_grid = False

        no_flat = True
        all_vertical = False
        flat_wall = False

        terrain_proportions = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.20, 0.20, 0.30, 0.30,
        ]

    class morphology:
        scaling_range = [0.6, 1.4]
        num_buckets = 100
        # PD gain correction: η = a*ξ³ + b*ξ² + c*ξ + d  (paper Eq.1)
        # Quadratic η=ξ² matches gravity-dominant parkour (τ_g ∝ m·r ∝ ξ²).
        # Linear [0,0,1,0] under-drives long legs; cubic may oscillate.
        pd_correction_coeffs = [0.03499, -0.3338, 1.382, -0.1001]  # η = ξ²
        target_morphology = None

    class parkour:
        gap_width = 1.0
        platform_height = 0.55
        waypoint_update_freq = 0.5
        forward_speed_target = 2.0

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 6.
        heading_command = True
        curriculum = False
        lin_vel_clip = 0.2
        ang_vel_clip = 0.4

        class ranges:
            lin_vel_x = [0.3, 0.8]       # forward speed cap for tracking_goal_vel (matches extreme-parkour max_ranges)
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0, 0]

        class max_ranges:
            lin_vel_x = [0.3, 0.8]       # same as ranges since curriculum=False
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-1.6, 1.6]

        class crclm_incremnt:
            lin_vel_x = 0.1
            lin_vel_y = 0.1
            ang_vel_yaw = 0.1
            heading = 0.5

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
        stiffness = {'HipX': 40.0, 'HipY': 40.0, 'Knee': 40.0}
        damping = {'HipX': 0.7, 'HipY': 0.7, 'Knee': 0.7}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/parkour_quadruped/urdf/parkour_quadruped.urdf'
        name = "parkour_quadruped"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "TORSO", "HIP"]
        terminate_after_contacts_on = []
        self_collisions = 0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 2.0]
        randomize_base_mass = True
        added_mass_range = [0.0, 3.0]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
        spatial_domain_rand = True
        motor_strength_range = [0.8, 1.2]
        action_buf_len = 8

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.20
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.4
        base_height_target = 0.25
        max_contact_force = 40.0

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
        add_noise = False
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.02

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
        scan_encoder_dims = [128, 64, 32]       # mt → scan_latent
        priv_encoder_dims = [64, 20]             # et → priv_latent (teacher)
        history_encoder_output_dim = 20          # ht → hist_latent (student, matches priv output)

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2.e-4
        schedule = 'adaptive'
        gamma = 0.98
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
        estimator_hidden_dims = [128, 64]       # xt → predicted_e't
        train_with_estimated_states = True

    class runner:
        policy_class_name = 'ParkourActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 30000
        save_interval = 100
        experiment_name = 'parkour_pretrain'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
