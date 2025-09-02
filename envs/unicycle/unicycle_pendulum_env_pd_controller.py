# envs/unicycle/unicycle_pendulum_env_pd_controller.py

import os
from typing import Any, Dict, Union

import numpy as np

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
| 17  | PD roll torque (about x‑axis)        | -1    | 1     | normalized torque        |
| 18  | PD yaw torque (about z‑axis)         | -1    | 1     | normalized torque        |

The observation is an `ndarray` with shape `(19,)`.
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
        "render_fps": 200,
    }

    def __init__(
        self,
        xml_file: str | None = None,
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int, np.ndarray]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01, # 0.005 s in XML × frame_skip
        max_steps: int = 1500,
        kp: float = 40.0,
        kd: float = 6.0,
        finish_line_x: float = 5.0,
        reward_weights: Dict[str, float] | None = None,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(
                os.path.dirname(__file__), "assets", "unicycle_pendulum_3d.xml"
            )

        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            reset_noise_scale,
            max_steps,
            kp,
            kd,
            finish_line_x,
            reward_weights,
        )

        obs_low  = np.full((19,), -1e10, dtype=np.float32)
        obs_high = np.full((19,), 1e10, dtype=np.float32)
        observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._reset_noise_scale = float(reset_noise_scale)
        self.max_steps = int(max_steps)
        self.finish_line_x = float(finish_line_x)
        self._kp = float(kp)
        self._kd = float(kd)

        self.steps = 0
        self.num_steps = 0
        self.num_episodes = 0
        self.num_success = 0
        self.num_falls = 0

        self.start_x_pos = 0.0
        self.prev_x_pos = 0.0
        self.goal_reached = False
        self.is_fallen = False
        self.ep_return = 0.0
        
        default_weights = {
            "progress": 35.0, "progress_exp_k": 0.30,
            "posture": 0.05, "pendulum": 0.5,
            "alive_bonus": 0.02, "goal_bonus": 100.0, "fall_penalty": -2.0
        }
        if reward_weights:
            default_weights.update(reward_weights)

        self.reward_weight_progress = default_weights["progress"]
        self.reward_progress_exp_k = default_weights["progress_exp_k"]
        self.reward_weight_posture = default_weights["posture"]
        self.reward_weight_pendulum = default_weights["pendulum"]
        self.reward_weight_alive_bonus = default_weights["alive_bonus"]
        self.reward_weight_goal_bonus = default_weights["goal_bonus"]
        self.reward_weight_fall_penalty = default_weights["fall_penalty"]

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        self.action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.last_info: Dict[str, Any] | None = None

    def step(self, action: np.ndarray):
        current_base_obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32, copy=False)
        
        a = np.asarray(action, dtype=np.float32).ravel()
        wheel_tau = float(a[0]) if a.size > 0 else 0.0
        pitch_tau = float(a[1]) if a.size >= 2 else 0.0

        roll_tau, yaw_tau = self.calculate_pd_torques(current_base_obs)
        full_action = np.array([wheel_tau, roll_tau, pitch_tau, yaw_tau], dtype=np.float32)

        self.do_simulation(full_action, self.frame_skip)
        self.steps += 1
        self.num_steps += 1

        obs = self.get_obs()
        x_pos, y_pos = float(obs[0]), float(obs[1])
        euler = self.get_euler_angles(obs)

        if not self.is_fallen and self.check_fall(euler):
            self.num_falls += 1
            self.is_fallen = True

        reward, rcomps = self.calculate_reward(x_pos, euler, obs)
        self.ep_return += reward
        self.update_success_count(x_pos)

        terminated = bool(self.is_fallen or x_pos >= self.finish_line_x)
        truncated = bool(self.max_steps and self.steps >= self.max_steps)

        info = self.get_info(x_pos, y_pos, euler, reward, rcomps)
        self.last_info = dict(info)
        if self.render_mode == "human":
            self.render()

        obs = self.get_obs().astype(np.float32, copy=False)
        return obs, float(reward), terminated, truncated, info

    def reset_model(self):
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale

        qpos = (self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )).astype(np.float32)
        qvel = (self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )).astype(np.float32)

        quat = qpos[3:7]
        qpos[3:7] = quat / (np.linalg.norm(quat) + 1e-8)

        self.set_state(qpos, qvel)

        self.steps = 0
        self.start_x_pos = float(qpos[0])
        self.prev_x_pos = self.start_x_pos
        self.goal_reached = False
        self.is_fallen = False
        self.num_episodes += 1
        self.ep_return = 0.0
        self.last_info = None

        initial_obs = self.get_obs().astype(np.float32, copy=False)
        return initial_obs

    def get_obs(self) -> np.ndarray:
        base_obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32, copy=False)
        pd_torques = self.calculate_pd_torques(base_obs)
        return np.concatenate([base_obs, pd_torques])

    def _get_obs(self) -> np.ndarray:
        return self.get_obs()
    
    def get_euler_angles(self, observation: np.ndarray) -> np.ndarray:
        w, x, y, z = map(float, observation[3:7])
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return np.array([roll, pitch, yaw], dtype=np.float32)
    
    def calculate_pd_torques(self, observation: np.ndarray) -> np.ndarray:
        roll, _, yaw = self.get_euler_angles(observation)
        roll_vel, _, yaw_vel = observation[12:15]
        roll_tau = np.clip(-(self._kp * roll + self._kd * roll_vel) / 10.0, -1.0, 1.0)
        yaw_tau = np.clip(-(self._kp * yaw  + self._kd * yaw_vel) / 10.0, -1.0, 1.0)
        return np.array([roll_tau, yaw_tau], dtype=np.float32)

    def calculate_reward(self, x_pos: float, euler: np.ndarray, obs: np.ndarray):
        # 1) Progress
        dx = x_pos - self.prev_x_pos
        k = float(self.reward_progress_exp_k)
        base_arg = np.clip(k * (self.prev_x_pos - self.start_x_pos), -60.0, 60.0)
        growth_arg = np.clip(k * dx, -10.0, 10.0)
        exp_progress = np.exp(base_arg) * np.expm1(growth_arg)
        progress_reward = self.reward_weight_progress * float(exp_progress)

        # 2) Pose
        roll, pitch, _ = euler
        pend = float(obs[7])
        upright_penalty = self.reward_weight_posture * (
            roll*roll + pitch*pitch + self.reward_weight_pendulum * pend*pend
        )
        posture_reward = -upright_penalty

        # 3) Event
        crossed = (self.prev_x_pos < self.finish_line_x <= x_pos)
        goal_bonus = self.reward_weight_goal_bonus if crossed else 0.0
        alive_bonus = self.reward_weight_alive_bonus if not self.is_fallen else 0.0
        fall_penalty = self.reward_weight_fall_penalty if self.is_fallen else 0.0

        reward = progress_reward + posture_reward + alive_bonus + goal_bonus + fall_penalty

        comps = {
            "reward_progress": progress_reward,
            "reward_posture": posture_reward,
            "reward_alive_bonus": alive_bonus,
            "reward_goal_bonus": goal_bonus,
            "reward_fall_penalty": fall_penalty,
        }
        return float(reward), comps

    def check_fall(self, euler: np.ndarray) -> bool:
        roll, pitch, _ = euler
        pendulum_ang = float(self.data.qpos[7])
        limit = np.pi / 3.0
        return (abs(roll) > limit) or (abs(pitch) > limit) or (abs(pendulum_ang) > limit)

    def update_success_count(self, x_pos: float) -> None:
        if self.prev_x_pos < self.finish_line_x <= x_pos:
            self.num_success += 1
            self.goal_reached = True
        self.prev_x_pos = x_pos

    def get_info(self, x_pos: float, y_pos: float, euler: np.ndarray, reward: float, rcomps: Dict[str, float] | None = None) -> Dict[str, Any]:
        roll, pitch, yaw = euler
        info: Dict[str, Any] = {
            "x_position": x_pos, 
            "y_position": y_pos, 
            "finish_line_x": float(self.finish_line_x),
            "goal_reached": self.goal_reached,
            "is_fallen": self.is_fallen, 
            "unicycle_roll": roll, 
            "unicycle_pitch": pitch,
            "unicycle_yaw": yaw, 
            "pendulum_ang": float(self.data.qpos[7]), 
            "steps": self.steps,
            "num_steps": self.num_steps, 
            "num_episodes": self.num_episodes, 
            "num_success": self.num_success,
            "num_falls": self.num_falls,
            "reward": float(reward), 
            "episode_return": float(self.ep_return),
        }
        if rcomps:
            info.update(rcomps)
        return info