# envs/unicycle/unicycle_env_pd_controller.py

import os
from typing import Dict, Union

import numpy as np
from scipy.spatial.transform import Rotation

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


"""
The agent takes a 2‑element vector for actions.

| Num | Action                                 | Control Range | Name (in XML)           | Joint         | Gear Factor | Effective Torque (N·m)   |
|-----|----------------------------------------|---------------|-------------------------|---------------|------------ |--------------------------|
| 0   | Torque applied on the wheel            | -1 to 1       | wheel_motor             | wheel_joint   | 25          | -25 to 25                |
| 1   | Pitch control torque (about y‑axis)    | -1 to 1       | pitch_control_motor     | free_joint    | 10          | -10 to 10                |

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
| 7   | angle of the pendulum                | -Inf  | Inf   | angle (rad)              |
| 8   | angle of the wheel                   | -Inf  | Inf   | angle (rad)              |
| 9   | velocity in the x‑direction          | -Inf  | Inf   | velocity (m/s)           |
| 10  | velocity in the y‑direction          | -Inf  | Inf   | velocity (m/s)           |
| 11  | velocity in the z‑direction          | -Inf  | Inf   | velocity (m/s)           |
| 12  | angular velocity about x‑axis        | -Inf  | Inf   | angular velocity (rad/s) |
| 13  | angular velocity about y‑axis        | -Inf  | Inf   | angular velocity (rad/s) |
| 14  | angular velocity about z‑axis        | -Inf  | Inf   | angular velocity (rad/s) |
| 15  | angular velocity of the pendulum     | -Inf  | Inf   | angular velocity (rad/s) |
| 16  | angular velocity of the wheel        | -Inf  | Inf   | angular velocity (rad/s) |

The observation is an `ndarray` with shape `(17,)`.
"""

# Default camera configuration
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
}

class UnicyclePendulumPD(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 200
    }

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 1,  # 0.005 s in XML × frame_skip
        default_camera_config: Dict[str, Union[float, int, np.ndarray]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        max_steps: int = 1500,

        # PD stabilizer gains
        kp: float = 40.0,
        kd: float = 6.0,

        # Task / termination
        goal_x: float = 5.0,
        track_half_width: float = 4.0,

        # Rewards
        r_goal: float = 20.0, # One-time terminal bonus on goal crossing
        k_shape: float = 1.0, # Progress shaping coefficient (per meter)
        c_time: float = 1.0, # Per-second time penalty
        c_tilt: float = 0.5, # Per-second tilt penalty coefficient (rad^2)
        c_lat: float = 0.1, # Per-second lateral penalty coefficient (m^2)
        c_ctrl: float = 0.01, # Per-second control penalty (2 agent actions only)
        fall_penalty: float = 20.0, # One-time penalty if fallen
        reward_scale: float = 50.0,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "unicycle_pendulum_3d.xml")

        utils.EzPickle.__init__(
            self, xml_file, frame_skip, reset_noise_scale, max_steps,
            kp, kd, goal_x, track_half_width,
            r_goal, k_shape, c_time, c_tilt, c_lat, c_ctrl, fall_penalty, reward_scale, **kwargs
        )

        # Observation / action spaces
        observation_space = Box(low=-1e10, high=1e10, shape=(17,), dtype=np.float32)
        action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = observation_space
        self.action_space = action_space

        # Episode / task state
        self._reset_noise_scale = float(reset_noise_scale)
        self.max_steps = int(max_steps)
        self.goal_x = float(goal_x)
        self.track_half_width = float(track_half_width)

        self.success_distance = self.goal_x

        # PD gains
        self._kp = float(kp)
        self._kd = float(kd)

        # Reward parameters
        self.r_goal = float(r_goal)
        self.k_shape = float(k_shape)
        self.c_time = float(c_time)
        self.c_tilt = float(c_tilt)
        self.c_lat = float(c_lat)
        self.c_ctrl = float(c_ctrl)
        self.fall_penalty = float(fall_penalty)
        self.reward_scale = float(reward_scale)

        # Bookkeeping counters
        self.steps = 0 # Steps within the current episode
        self.num_steps = 0 # Total steps across episodes
        self.num_episodes = 0
        self.num_success = 0
        self.num_falls = 0

        # Flags & internals
        self.prev_x_pos = 0.0
        self.goal_reached = False
        self.line_crossed = False
        self.is_fallen = False
        self.prev_dist = None # For telescoping progress shaping
        self.last_agent_action = np.zeros(2, dtype=np.float32)  # for control penalty

        # Initialize MuJoCo base
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

    def step(self, action):
        obs_now = self.get_obs()
        roll, pitch, yaw = self.get_euler_angles(obs_now)
        roll_vel, pitch_vel, yaw_vel = obs_now[12:15]

        action_array = np.asarray(action, dtype=np.float32)
        if len(action_array) >= 2:
            wheel_tau, pitch_tau = action_array[:2]
        else:
            wheel_tau = action_array[0] if len(action_array) > 0 else 0.0
            pitch_tau = 0.0
        self.last_agent_action = np.array([wheel_tau, pitch_tau], dtype=np.float32)

        kp, kd = self._kp, self._kd
        roll_tau = np.clip(-(kp * roll + kd * roll_vel) / 10.0, -1.0, 1.0)
        yaw_tau  = np.clip(-(kp * yaw  + kd * yaw_vel)  / 10.0, -1.0, 1.0)

        full_action = np.array([wheel_tau, roll_tau, pitch_tau, yaw_tau], dtype=np.float32)

        self.do_simulation(full_action, self.frame_skip)
        self.steps += 1
        self.num_steps += 1

        obs = self.get_obs()
        x_pos, y_pos = float(obs[0]), float(obs[1])
        euler = self.get_euler_angles(obs)

        fallen_before = self.is_fallen
        if not self.is_fallen and self._check_fall(euler):
            self.num_falls += 1
            self.is_fallen = True

        reward_unscaled, rcomps = self.calculate_reward(x_pos, y_pos, euler, obs)

        if self.is_fallen and not fallen_before:
            reward_unscaled -= self.fall_penalty

        reward = reward_unscaled * self.reward_scale

        self.update_success_count(x_pos)

        terminated = bool(
            self.is_fallen or
            abs(y_pos) > self.track_half_width or
            x_pos >= self.goal_x
        )
        truncated = bool(self.max_steps and self.steps >= self.max_steps)

        info = self.get_info(x_pos, y_pos, euler, reward, rcomps)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale

        qpos = (self.init_qpos + self.np_random.uniform(size=self.model.nq, low=noise_low, high=noise_high)).astype(np.float32)
        qvel = (self.init_qvel + self.np_random.uniform(size=self.model.nv, low=noise_low, high=noise_high)).astype(np.float32)

        quat = qpos[3:7]
        qpos[3:7] = quat / np.linalg.norm(quat + 1e-8)

        self.set_state(qpos, qvel)

        self.steps = 0
        self.prev_x_pos = float(qpos[0])
        self.goal_reached = False
        self.is_fallen = False
        self.line_crossed = False
        self.num_episodes += 1

        x0 = float(qpos[0])
        self.prev_dist = max(self.goal_x - x0, 0.0)

        self.last_agent_action = np.zeros(2, dtype=np.float32)

        return self.get_obs()


    def get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32, copy=False)

    def get_euler_angles(self, observation: np.ndarray) -> np.ndarray:
        quat_wxyz = observation[3:7] # MuJoCo stores (w, x, y, z)
        quat_xyzw = np.roll(quat_wxyz, -1) # SciPy expects (x, y, z, w)
        return Rotation.from_quat(quat_xyzw).as_euler("xyz") # radians

    def calculate_reward(self, x_pos: float, y_pos: float, euler: np.ndarray, obs: np.ndarray):
        roll, pitch, _ = euler
        pendulum_ang = float(obs[7])
        dt = float(self.dt)

        dist = max(self.goal_x - float(x_pos), 0.0)
        r_progress = self.k_shape * (self.prev_dist - dist)
        self.prev_dist = dist

        r_time = -self.c_time * dt

        tilt_sq = roll * roll + pitch * pitch + pendulum_ang * pendulum_ang
        r_stability = -self.c_tilt * tilt_sq * dt
        r_lateral   = -self.c_lat  * (float(y_pos) ** 2) * dt
        a = getattr(self, "last_agent_action", np.zeros(2, dtype=np.float32))
        r_control   = -self.c_ctrl * float(np.dot(a, a)) * dt

        r_goal = 0.0
        if (not self.line_crossed) and (x_pos >= self.goal_x):
            r_goal = self.r_goal
            self.line_crossed = True

        reward_unscaled = r_progress + r_time + r_stability + r_lateral + r_control + r_goal

        comps = {
            "reward_progress": r_progress,
            "penalty_time": r_time,
            "penalty_stability": r_stability,
            "penalty_lateral": r_lateral,
            "penalty_control": r_control,
            "reward_goal": r_goal,
            "dist_to_goal": dist,
            "reward_unscaled": reward_unscaled,
        }
        return reward_unscaled, comps

    def _check_fall(self, euler: np.ndarray) -> bool:
        roll, pitch, _ = euler
        pendulum_ang = float(self.data.qpos[7])
        limit = np.pi / 3.0
        return (abs(roll) > limit) or (abs(pitch) > limit) or (abs(pendulum_ang) > limit)

    def update_success_count(self, x_pos: float) -> None:
        if self.prev_x_pos < self.goal_x <= x_pos:
            self.num_success += 1
            self.goal_reached = True
        self.prev_x_pos = x_pos

    def get_info(self, x_pos: float, y_pos: float, euler: np.ndarray, reward: float, rcomps=None) -> Dict:
        roll, pitch, _ = euler
        info = {
            # task state
            "x_position": x_pos,
            "y_position": y_pos,
            "goal_x": self.goal_x,
            "goal_reached": self.goal_reached,

            # attitude
            "unicycle_roll": roll,
            "unicycle_pitch": pitch,
            "pendulum_ang": float(self.data.qpos[7]),

            # episode stats
            "steps": self.steps,
            "num_steps": self.num_steps,
            "num_success": self.num_success,
            "num_episodes": self.num_episodes,
            "num_falls": self.num_falls,
            "is_fallen": self.is_fallen,
            
            # reward
            "reward": float(reward),
        }
        if rcomps is not None:
            info.update(rcomps)
        return info