"""Parkour terrain with waypoint system — references extreme-parkour design.

Terrain types (index → name):
  0: smooth slope up      7: stepping stones
  1: smooth slope down    8: gaps
  2: rough slope up       9: smooth flat
  3: rough slope down     10: pit
  4: rough stairs up      11: wall
  5: rough stairs down    12: platform
  6: discrete             13: large stairs up
                          14: large stairs down
                          15: parkour (stepping+incline)
                          16: parkour_hurdle
                          17: parkour_flat
                          18: parkour_step (high jump)
                          19: parkour_gap (long jump)

Each parkour terrain generates waypoints embedded in the terrain.
"""

import numpy as np
import random
from isaacgym import terrain_utils
from legged_gym.utils.terrain import Terrain


class ParkourHeightField(Terrain):
    def __init__(self, cfg, num_robots):
        self.parkour_cfg = cfg.parkour
        raw_props = np.array(cfg.terrain.terrain_proportions)
        raw_props = raw_props / np.sum(raw_props)
        self.proportions = [np.sum(raw_props[:i+1]) for i in range(len(raw_props))]

        self.goals = np.zeros((cfg.terrain.num_rows, cfg.terrain.num_cols, cfg.terrain.num_goals, 3))
        self.num_goals = cfg.terrain.num_goals
        self.terrain_type = np.zeros((cfg.terrain.num_rows, cfg.terrain.num_cols))
        super().__init__(cfg.terrain, num_robots)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )

        slope = difficulty * 0.4
        step_height = 0.02 + 0.14 * difficulty
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty

        # --- Non-parkour terrains (indices 0-13, ref extreme-parkour) ---

        if choice < self.proportions[0]:
            idx = 0
            if choice < self.proportions[0] / 2:
                idx = 1
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)

        elif choice < self.proportions[2]:
            idx = 2
            if choice < self.proportions[1]:
                idx = 3
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[4]:
            idx = 4
            if choice < self.proportions[3]:
                idx = 5
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[5]:
            idx = 6
            terrain_utils.discrete_obstacles_terrain(
                terrain, discrete_obstacles_height,
                rectangle_min_size=0.5, rectangle_max_size=2.,
                num_rectangles=20, platform_size=3.
            )
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[6]:
            idx = 7
            stones_size = 1.5 - 1.2 * difficulty
            stepping_stones_terrain(terrain, stone_size=1.5 - 0.2 * difficulty,
                                   stone_distance=0.0 + 0.4 * difficulty,
                                   max_height=0.2 * difficulty, platform_size=1.2)
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[7]:
            idx = 8
            gap_parkour_terrain(terrain, difficulty, platform_size=4)
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[8]:
            idx = 9
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[9]:
            idx = 10
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        elif choice < self.proportions[10]:
            idx = 11
            if self.cfg.all_vertical:
                half_slope_difficulty = 1.0
            else:
                difficulty_wall = difficulty * 1.3
                if not self.cfg.no_flat:
                    difficulty_wall -= 0.1
                if difficulty_wall > 1:
                    half_slope_difficulty = 1.0
                elif difficulty_wall < 0:
                    self.add_roughness(terrain)
                    terrain.slope_vector = np.array([1, 0., 0]).astype(np.float32)
                    terrain.idx = idx
                    return terrain
                else:
                    half_slope_difficulty = difficulty_wall
            wall_width = 4 - half_slope_difficulty * 4
            if self.cfg.flat_wall:
                half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            else:
                half_sloped_terrain(terrain, wall_width=wall_width, start2center=0.5, max_height=1.5)
            max_height = terrain.height_field_raw.max()
            top_mask = terrain.height_field_raw > max_height - 0.05
            self.add_roughness(terrain, difficulty=1)
            terrain.height_field_raw[top_mask] = max_height

        elif choice < self.proportions[11]:
            idx = 12
            half_platform_terrain(terrain, max_height=0.1 + 0.4 * difficulty)
            self.add_roughness(terrain, difficulty=1)

        elif choice < self.proportions[13]:
            idx = 13
            height = 0.1 + 0.3 * difficulty
            if choice < self.proportions[12]:
                idx = 14
                height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=1., step_height=height, platform_size=3.)
            self.add_roughness(terrain, difficulty)

        # --- Parkour terrains (indices 15-19, ref extreme-parkour, scaled for 8m terrain) ---
        # num_stones = num_goals - 2 = 4 (vs extreme-parkour 6 for 18m terrain)

        elif choice < self.proportions[14]:
            idx = 15
            x_range = [-0.1, 0.1 + 0.3 * difficulty]
            y_range = [0.2, 0.3 + 0.1 * difficulty]
            stone_len = [0.9 - 0.3 * difficulty, 1 - 0.2 * difficulty]
            incline_height = 0.25 * difficulty
            last_incline_height = incline_height + 0.1 - 0.1 * difficulty
            parkour_terrain(terrain,
                            num_stones=self.num_goals - 2,
                            x_range=x_range, y_range=y_range,
                            incline_height=incline_height,
                            stone_len=stone_len, stone_width=1.0,
                            last_incline_height=last_incline_height,
                            pad_height=0, pit_depth=[0.2, 1])
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[15]:
            idx = 16
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1 + 0.3 * difficulty,
                                   hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.25 * difficulty],
                                   pad_height=0,
                                   x_range=[1.2, 2.2],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.4, 0.8])
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[16]:
            idx = 17
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1 + 0.3 * difficulty,
                                   hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty],
                                   pad_height=0,
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.45, 1],
                                   flat=True)
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[17]:
            idx = 18
            parkour_step_terrain(terrain,
                                 num_stones=self.num_goals - 2,
                                 step_height=0.1 + 0.35 * difficulty,
                                 x_range=[0.3, 1.5],
                                 y_range=self.cfg.y_range,
                                 half_valid_width=[0.5, 1],
                                 pad_height=0)
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[18]:
            idx = 19
            parkour_gap_terrain(terrain,
                                num_gaps=self.num_goals - 2,
                                gap_size=0.1 + 0.7 * difficulty,
                                gap_depth=[0.2, 1],
                                pad_height=0,
                                x_range=[0.8, 1.5],
                                y_range=self.cfg.y_range,
                                half_valid_width=[0.6, 1.2])
            self.add_roughness(terrain, difficulty)

        else:
            idx = 20
            demo_terrain(terrain)
            self.add_roughness(terrain, difficulty)

        terrain.idx = idx
        return terrain

    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(
            terrain, min_height=-height, max_height=height,
            step=0.005, downsampled_scale=self.cfg.downsampled_scale
        )

    def add_terrain_to_map(self, terrain, row, col):
        i, j = row, col
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        if getattr(self.cfg, 'origin_zero_z', True):
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx

        if hasattr(terrain, 'goals'):
            n_goals = min(terrain.goals.shape[0], self.goals.shape[2])
            self.goals[i, j, :n_goals, :2] = terrain.goals[:n_goals] + [i * self.env_length, j * self.env_width]
        else:
            for g in range(self.num_goals):
                progress = (g + 1) / (self.num_goals + 1 + 1e-6)
                self.goals[i, j, g] = [
                    env_origin_x + progress * (self.env_length - 2),
                    env_origin_y, 0
                ]


# ==================== Parkour terrain generators (from extreme-parkour) ====================

def parkour_terrain(terrain, platform_len=2.5, platform_height=0., num_stones=4,
                    x_range=None, y_range=None, z_range=None,
                    stone_len=1.0, stone_width=0.6, pad_width=0.1, pad_height=0.5,
                    incline_height=0.1, last_incline_height=0.6, last_stone_len=1.6,
                    pit_depth=None):
    """Stepping stones with alternating left/right inclines. Ref: extreme-parkour."""
    if x_range is None: x_range = [1.8, 1.9]
    if y_range is None: y_range = [0., 0.1]
    if z_range is None: z_range = [-0.2, 0.2]
    if pit_depth is None: pit_depth = [0.5, 1.]

    goals = np.zeros((num_stones + 2, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)

    mid_y = terrain.length // 2
    stone_len_val = np.random.uniform(*stone_len)
    stone_len_val = 2 * round(stone_len_val / 2.0, 1)
    stone_len_px = round(stone_len_val / terrain.horizontal_scale)
    dis_x_min = stone_len_px + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len_px + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_px = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len_px, :] = platform_height_px

    stone_width_px = round(stone_width / terrain.horizontal_scale)
    last_stone_len_px = round(last_stone_len / terrain.horizontal_scale)
    incline_height_px = round(incline_height / terrain.vertical_scale)
    last_incline_height_px = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len_px - np.random.randint(dis_x_min, dis_x_max) + stone_len_px // 2
    goals[0] = [platform_len_px - stone_len_px // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)

    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2 * (left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len_px // 4
            heights = np.tile(np.linspace(-last_incline_height_px, last_incline_height_px, stone_width_px),
                            (last_stone_len_px, 1)) * pos_neg
            terrain.height_field_raw[dis_x - last_stone_len_px // 2:dis_x + last_stone_len_px // 2,
                                     dis_y - stone_width_px // 2:dis_y + stone_width_px // 2] = heights.astype(int)
        else:
            heights = np.tile(np.linspace(-incline_height_px, incline_height_px, stone_width_px),
                            (stone_len_px, 1)) * pos_neg
            terrain.height_field_raw[dis_x - stone_len_px // 2:dis_x + stone_len_px // 2,
                                     dis_y - stone_width_px // 2:dis_y + stone_width_px // 2] = heights.astype(int)
        goals[i + 1] = [dis_x, dis_y]
        left_right_flag = 1 - left_right_flag

    final_dis_x = dis_x + 2 * np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = dis_x + last_stone_len_px // 2 + round(0.05 / terrain.horizontal_scale)
    terrain.height_field_raw[final_platform_start:, :] = platform_height_px
    goals[-1] = [final_dis_x, mid_y]
    terrain.goals = goals * terrain.horizontal_scale

    pad_px = int(pad_width / terrain.horizontal_scale)
    pad_h_px = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_px] = pad_h_px
    terrain.height_field_raw[:, -pad_px:] = pad_h_px
    terrain.height_field_raw[:pad_px, :] = pad_h_px
    terrain.height_field_raw[-pad_px:, :] = pad_h_px


def parkour_hurdle_terrain(terrain, platform_len=2.5, platform_height=0., num_stones=8,
                           stone_len=0.3, x_range=None, y_range=None,
                           half_valid_width=None, hurdle_height_range=None,
                           pad_width=0.1, pad_height=0.5, flat=False):
    """Hurdle terrain — robots jump over barriers. Ref: extreme-parkour."""
    if x_range is None: x_range = [1.5, 2.4]
    if y_range is None: y_range = [-0.4, 0.4]
    if half_valid_width is None: half_valid_width = [0.4, 0.8]
    if hurdle_height_range is None: hurdle_height_range = [0.2, 0.3]

    goals = np.zeros((num_stones + 2, 2))
    mid_y = terrain.length // 2

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    half_valid_width_px = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_px = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len_px, :] = platform_height_px
    stone_len_px = round(stone_len / terrain.horizontal_scale)

    dis_x = platform_len_px
    goals[0] = [platform_len_px - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[dis_x - stone_len_px // 2:dis_x + stone_len_px // 2, :] = \
                np.random.randint(hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[dis_x - stone_len_px // 2:dis_x + stone_len_px // 2,
                                     :mid_y + rand_y - half_valid_width_px] = 0
            terrain.height_field_raw[dis_x - stone_len_px // 2:dis_x + stone_len_px // 2,
                                     mid_y + rand_y + half_valid_width_px:] = 0
        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]

    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = int(terrain.width - 0.5 / terrain.horizontal_scale)
    goals[-1] = [final_dis_x, mid_y]
    terrain.goals = goals * terrain.horizontal_scale

    pad_px = int(pad_width / terrain.horizontal_scale)
    pad_h_px = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_px] = pad_h_px
    terrain.height_field_raw[:, -pad_px:] = pad_h_px
    terrain.height_field_raw[:pad_px, :] = pad_h_px
    terrain.height_field_raw[-pad_px:, :] = pad_h_px


def parkour_gap_terrain(terrain, platform_len=2.5, platform_height=0., num_gaps=8, gap_size=0.3,
                        x_range=None, y_range=None, half_valid_width=None, gap_depth=None,
                        pad_width=0.1, pad_height=0.5, flat=False):
    """Long jump terrain with waypoints. Ref: extreme-parkour."""
    if x_range is None: x_range = [1.6, 2.4]
    if y_range is None: y_range = [-1.2, 1.2]
    if half_valid_width is None: half_valid_width = [0.6, 1.2]
    if gap_depth is None: gap_depth = [0.2, 1]

    goals = np.zeros((num_gaps + 2, 2))
    mid_y = terrain.length // 2

    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_px = round(platform_height / terrain.vertical_scale)
    gap_depth_px = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
    half_valid_width_px = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    terrain.height_field_raw[0:platform_len_px, :] = platform_height_px

    gap_size_px = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size_px
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size_px

    dis_x = platform_len_px
    goals[0] = [platform_len_px - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        terrain.height_field_raw[dis_x - gap_size_px // 2:dis_x + gap_size_px // 2, :] = gap_depth_px
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y + rand_y - half_valid_width_px] = gap_depth_px
        terrain.height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width_px:] = gap_depth_px
        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]

    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = int(terrain.width - 0.5 / terrain.horizontal_scale)
    goals[-1] = [final_dis_x, mid_y]
    terrain.goals = goals * terrain.horizontal_scale

    pad_px = int(pad_width / terrain.horizontal_scale)
    pad_h_px = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_px] = pad_h_px
    terrain.height_field_raw[:, -pad_px:] = pad_h_px
    terrain.height_field_raw[:pad_px, :] = pad_h_px
    terrain.height_field_raw[-pad_px:, :] = pad_h_px


def parkour_step_terrain(terrain, platform_len=2.5, platform_height=0., num_stones=8,
                         x_range=None, y_range=None, half_valid_width=None, step_height=0.2,
                         pad_width=0.1, pad_height=0.5):
    """High jump / step terrain with waypoints. Ref: extreme-parkour."""
    if x_range is None: x_range = [0.2, 0.4]
    if y_range is None: y_range = [-0.15, 0.15]
    if half_valid_width is None: half_valid_width = [0.45, 0.5]

    goals = np.zeros((num_stones + 2, 2))
    mid_y = terrain.length // 2

    dis_x_min = round((x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round((x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    step_height_px = round(step_height / terrain.vertical_scale)
    half_valid_width_px = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_px = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len_px, :] = platform_height_px

    dis_x = platform_len_px
    last_dis_x = dis_x
    stair_height = 0
    goals[0] = [platform_len_px - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height_px
        elif i > num_stones // 2:
            stair_height -= step_height_px
        terrain.height_field_raw[dis_x:dis_x + rand_x, :] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y + rand_y - half_valid_width_px] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width_px:] = 0
        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]

    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = int(terrain.width - 0.5 / terrain.horizontal_scale)
    goals[-1] = [final_dis_x, mid_y]
    terrain.goals = goals * terrain.horizontal_scale

    pad_px = int(pad_width / terrain.horizontal_scale)
    pad_h_px = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_px] = pad_h_px
    terrain.height_field_raw[:, -pad_px:] = pad_h_px
    terrain.height_field_raw[:pad_px, :] = pad_h_px
    terrain.height_field_raw[-pad_px:, :] = pad_h_px


# ==================== Auxiliary terrain generators (from extreme-parkour) ====================

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-1):
    def get_rand_dis_int(scale):
        return np.random.randint(int(-scale / terrain.horizontal_scale + 1), int(scale / terrain.horizontal_scale))
    stone_size_px = int(stone_size / terrain.horizontal_scale)
    stone_distance_px = int(stone_distance / terrain.horizontal_scale)
    max_height_px = int(max_height / terrain.vertical_scale)
    platform_size_px = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height_px - 1, max_height_px, step=1)
    start_x, start_y = 0, 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size_px)
            start_x = np.random.randint(0, stone_size_px)
            stop_x = max(0, start_x - stone_distance_px - get_rand_dis_int(0.2))
            terrain.height_field_raw[0:stop_x, start_y:stop_y] = np.random.choice(height_range)
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size_px)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_x += stone_size_px + stone_distance_px + get_rand_dis_int(0.2)
            start_y += stone_size_px + stone_distance_px + get_rand_dis_int(0.2)
    else:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size_px)
            start_y = np.random.randint(0, stone_size_px)
            stop_y = max(0, start_y - stone_distance_px)
            terrain.height_field_raw[start_x:stop_x, 0:stop_y] = np.random.choice(height_range)
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size_px)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_y += stone_size_px + stone_distance_px
            start_x += stone_size_px + stone_distance_px
    x1, x2 = (terrain.width - platform_size_px) // 2, (terrain.width + platform_size_px) // 2
    y1, y2 = (terrain.length - platform_size_px) // 2, (terrain.length + platform_size_px) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0


def gap_parkour_terrain(terrain, difficulty, platform_size=2.):
    gap_size = 0.1 + 0.3 * difficulty
    gap_size_px = int(gap_size / terrain.horizontal_scale)
    platform_size_px = int(platform_size / terrain.horizontal_scale)
    center_x, center_y = terrain.length // 2, terrain.width // 2
    x1 = (terrain.length - platform_size_px) // 2
    x2 = x1 + gap_size_px
    y1 = (terrain.width - platform_size_px) // 2
    y2 = y1 + gap_size_px
    terrain.height_field_raw[center_x - x2:center_x + x2, center_y - y2:center_y + y2] = -400
    terrain.height_field_raw[center_x - x1:center_x + x1, center_y - y1:center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    depth_px = int(depth / terrain.vertical_scale)
    platform_size_px = int(platform_size / terrain.horizontal_scale / 2)
    x1, x2 = terrain.length // 2 - platform_size_px, terrain.length // 2 + platform_size_px
    y1, y2 = terrain.width // 2 - platform_size_px, terrain.width // 2 + platform_size_px
    terrain.height_field_raw[x1:x2, y1:y2] = -depth_px


def half_sloped_terrain(terrain, wall_width=4, start2center=0.7, max_height=1):
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array([wall_width_int * terrain.horizontal_scale, 0., max_height]).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)


def half_platform_terrain(terrain, start2center=2, max_height=1):
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain.height_field_raw[:, :] = max_height_int
    terrain.height_field_raw[-slope_start:slope_start, -slope_start:slope_start] = 0


def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth / 2, mid_y]
    terrain.height_field_raw[platform_length:platform_length + hurdle_depth,
                             round(mid_y - hurdle_width / 2):round(mid_y + hurdle_width / 2)] = hurdle_height
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length + first_step_depth / 2, mid_y]
    terrain.height_field_raw[platform_length:platform_length + first_step_depth,
                             round(mid_y - first_step_width / 2):round(mid_y + first_step_width / 2)] = first_step_height
    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    goals[2] = [platform_length + second_step_depth / 2, mid_y]
    terrain.height_field_raw[platform_length:platform_length + second_step_depth,
                             round(mid_y - first_step_width / 2):round(mid_y + first_step_width / 2)] = first_step_height
    platform_length += second_step_depth + round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length + third_step_depth / 2, mid_y]
    terrain.height_field_raw[platform_length:platform_length + third_step_depth,
                             round(mid_y - third_step_width / 2):round(mid_y + third_step_width / 2)] = first_step_height
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    goals[4] = [platform_length + forth_step_depth / 2, mid_y]
    terrain.height_field_raw[platform_length:platform_length + forth_step_depth,
                             round(mid_y - third_step_width / 2):round(mid_y + third_step_width / 2)] = first_step_height
    platform_length += forth_step_depth + round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    slope_height = round(np.random.uniform(0.15, 0.22) / terrain.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / terrain.horizontal_scale)
    slope_width = round(1.0 / terrain.horizontal_scale)
    platform_height = slope_height + np.random.randint(0, int(0.2 / terrain.vertical_scale))
    goals[5] = [platform_length + slope_depth / 2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    terrain.height_field_raw[platform_length:platform_length + slope_depth,
                             left_y - slope_width // 2:left_y + slope_width // 2] = heights.astype(int) + platform_height
    platform_length += slope_depth + round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    goals[6] = [platform_length + slope_depth / 2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    terrain.height_field_raw[platform_length:platform_length + slope_depth,
                             right_y - slope_width // 2:right_y + slope_width // 2] = heights.astype(int) + platform_height
    platform_length += slope_depth + round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale) + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale
