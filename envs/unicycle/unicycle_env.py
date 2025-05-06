# envs/unicycle/unicycle_env.py

import os
from typing import Dict, Union

import numpy as np
from scipy.spatial.transform import Rotation

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

###################################################
## Action Space
###################################################
"""
The agent takes a 4‑element vector for actions.

| Num | Action                                 | Control Range | Name (in XML)           | Joint         | Gear Factor | Effective Torque (N·m)   |
|-----|----------------------------------------|---------------|-------------------------|---------------|------------ |--------------------------|
| 0   | Torque applied on the wheel            | -1 to 1       | wheel_motor             | wheel_joint   | 25          | -25 to 25                |
| 1   | Roll control torque (about x‑axis)     | -1 to 1       | roll_control_motor      | free_joint    | 10          | -10 to 10                |
| 2   | Pitch control torque (about y‑axis)    | -1 to 1       | pitch_control_motor     | free_joint    | 10          | -10 to 10                |
| 3   | Yaw control torque  (about z‑axis)     | -1 to 1       | yaw_control_motor       | free_joint    | 10          | -10 to 10                |

Solution model will take action based on the environment after student's action taken.
"""

###################################################
## Observation Space
###################################################
"""
The observation consists of:

| Num | Observation                          | Min   | Max   | Type (Unit)              |
|-----|--------------------------------------|-------|-------|--------------------------|
| 0   | position in the x‑direction          | -Inf  | Inf   | position (m)             |
| 1   | position in the y‑direction          | -Inf  | Inf   | position (m)             |
| 2   | position in the z‑direction          | -Inf  | Inf   | position (m)             |
| 3   | w‑component of the orientation quat  | -1    | 1     | quaternion               |
| 4   | x‑component of the orientation quat  | -1    | 1     | quaternion               |
| 5   | y‑component of the orientation quat  | -1    | 1     | quaternion               |
| 6   | z‑component of the orientation quat  | -1    | 1     | quaternion               |
| 7   | angle of the wheel                   | -Inf  | Inf   | angle (rad)              |
| 8   | velocity in the x‑direction          | -Inf  | Inf   | velocity (m/s)           |
| 9   | velocity in the y‑direction          | -Inf  | Inf   | velocity (m/s)           |
| 10  | velocity in the z‑direction          | -Inf  | Inf   | velocity (m/s)           |
| 11  | angular velocity about x‑axis        | -Inf  | Inf   | angular velocity (rad/s) |
| 12  | angular velocity about y‑axis        | -Inf  | Inf   | angular velocity (rad/s) |
| 13  | angular velocity about z‑axis        | -Inf  | Inf   | angular velocity (rad/s) |
| 14  | angular velocity of the wheel        | -Inf  | Inf   | angular velocity (rad/s) |

The observation is an `ndarray` with shape `(15,)`.
"""

# Default camera configuration
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
}

class Unicycle(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 200        
    }

    # ------------------------------------------------------------------ #
    #                               INIT                                 #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 1,          # 0.005 s in XML × frame_skip
        default_camera_config: Dict[str, Union[float, int, np.ndarray]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        max_steps: int = 1500,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "unicycle_3d.xml")

        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, max_steps, **kwargs)

        obs_space = Box(low=-1e10, high=1e10, shape=(15,), dtype=np.float32)
        self.observation_space = obs_space

        self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Environment state counters
        self._reset_noise_scale = reset_noise_scale
        self.max_steps          = max_steps
        self.num_steps          = 0
        self.num_episodes       = 0
        self.num_success        = 0
        self.prev_x_pos         = 0.0
        self.target_success_count = 1000
        self.goal_distance      = 5.5
        self.success_distance   = 5.0
        self.goal_reached       = False
        self.num_falls          = 0
        self._is_fallen         = False
        self.line_crossed = False       # new flag for 5.5 m

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=obs_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

    # ------------------------------------------------------------------ #
    #                                STEP                                #
    # ------------------------------------------------------------------ #
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.steps     += 1
        self.num_steps += 1

        obs   = self._get_obs()
        x_pos, y_pos = obs[:2]

        euler = self._get_euler_angles(obs)

        # Mark fall event only once
        if not self._is_fallen and self._check_fall(euler):
            self.num_falls += 1
            self._is_fallen = True

        # Reward must be computed **before** prev_x_pos is updated
        reward = self._calculate_reward(x_pos, y_pos, euler, obs)

        # success counter / prev_x_pos update
        self._update_success_count(x_pos)

        terminated = self._check_termination(x_pos, y_pos, euler)
        truncated  = bool(self.steps >= self.max_steps)
        info       = self._get_info(x_pos, y_pos, euler, reward)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #                               RESET                                #
    # ------------------------------------------------------------------ #
    def reset_model(self):
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale

        qpos = (self.init_qpos + self.np_random.uniform(size=self.model.nq, low=noise_low, high=noise_high)).astype(np.float32)

        qvel = (self.init_qvel + self.np_random.uniform(size=self.model.nv, low=noise_low, high=noise_high)).astype(np.float32)

        # Normalize quaternion to avoid invalid orientation
        quat = qpos[3:7]
        qpos[3:7] = quat / np.linalg.norm(quat + 1e-8)

        self.set_state(qpos, qvel)

        # Reset episode counters
        self.steps        = 0
        self.prev_x_pos   = 0.0
        self.goal_reached = False
        self.num_episodes += 1
        self._is_fallen   = False

        self.line_crossed = False       # reset
        self._prev_potential = -self.goal_distance   # start at -5.5

        return self._get_obs()

    # ------------------------------------------------------------------ #
    #                      OBSERVATION HELPERS                           #
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32, copy=False)

    def _get_euler_angles(self, observation):
        quat = observation[3:7]
        return Rotation.from_quat(np.roll(quat, -1)).as_euler("xyz")

    # ------------------------------------------------------------------ #
    #                          REWARD FUNCTION                           #
    # ------------------------------------------------------------------ #
    def _calculate_reward(self, x_pos, y_pos, euler, obs):
        roll, pitch, yaw = euler
        x_vel, y_vel = obs[8], obs[9]
        roll_vel, pitch_vel, yaw_vel = obs[11], obs[12], obs[13]
        wheel_vel = obs[14]

        # ---------- 1. Upright gate ----------
        upright = np.exp(-4.0 * (roll**2 + pitch**2))   # smoother than before

        # ---------- 2. Distance shaping (only before the line) ----------
        dist_to_goal = self.goal_distance - x_pos           # >0 before line
        if dist_to_goal > 0:
            potential = -dist_to_goal                      # 0  →  -5.5
            shaping_reward = 4.0 * (potential - self._prev_potential) * upright
            self._prev_potential = potential
        else:
            shaping_reward = 0.0                            # after crossing

        # ---------- 3. Cruise‑speed band reward ----------
        if 0.6 <= x_vel <= 1.2:
            speed_reward = 0.3 * upright
        else:
            speed_reward = 0.0

        # ---------- 4. Stability penalties ----------
        tilt_pen   = -0.4 * (roll**2 + pitch**2)                # quadratic
        ang_pen    = -0.02 * (roll_vel**2 + pitch_vel**2)
        yaw_pen    = -0.02 * (yaw_vel**2 + 10.0 * yaw**2)
        lat_pen    = -0.1  * abs(y_vel)
        wheel_pen  = -0.005 * np.clip(wheel_vel, -15, 15)**2

        # ---------- 5. One‑time goal bonus ----------
        goal_bonus = 0.0
        if (not self.line_crossed) and (x_pos >= self.goal_distance):  
            goal_bonus       = 100.0
            self.line_crossed = True   # prevent double reward

        # ---------- 6. Total reward ----------
        reward = (shaping_reward + speed_reward + goal_bonus +
                  tilt_pen + ang_pen + yaw_pen + lat_pen + wheel_pen)

        return reward


    # ------------------------------------------------------------------ #
    #                        TERMINATION LOGIC                           #
    # ------------------------------------------------------------------ #
    def _check_termination(self, x_pos, y_pos, euler):
        return bool(
            abs(y_pos) > 4.0 or
            x_pos >= self.goal_distance or
            self.num_success >= self.target_success_count or
            self._check_fall(euler)
        )

    def _check_fall(self, euler):
        roll, pitch, _ = euler
        # Fall when roll or pitch exceeds 60 deg (π/3)
        return abs(roll) > np.pi / 3 or abs(pitch) > np.pi / 3

    # ------------------------------------------------------------------ #
    #                       SUCCESS COUNTER                              #
    # ------------------------------------------------------------------ #
    def _update_success_count(self, x_pos: float) -> None:
        if self.prev_x_pos < self.success_distance <= x_pos:
            self.num_success += 1
            self.goal_reached = True
        self.prev_x_pos = x_pos

    # ------------------------------------------------------------------ #
    #                              INFO                                  #
    # ------------------------------------------------------------------ #
    def _get_info(self, x_pos, y_pos, euler, reward):
        roll, pitch, _ = euler
        return {
            "unicycle_roll":  roll,
            "unicycle_pitch": pitch,
            "x_position":     x_pos,
            "y_position":     y_pos,
            "goal_reached":   self.goal_reached,
            "steps":          self.steps,
            "num_steps":      self.num_steps,
            "num_success":    self.num_success,
            "num_episodes":   self.num_episodes,
            "num_falls":      self.num_falls,
            "is_fallen":      self._is_fallen,
            "reward":         reward,
        }
