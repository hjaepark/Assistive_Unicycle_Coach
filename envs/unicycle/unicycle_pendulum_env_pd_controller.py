# envs/unicycle/unicycle_env_pd_controller.py

import os
from typing import Any, Dict, Union

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
        "render_fps": 200,
    }

    def __init__(
        self,
        xml_file: str | None = None,
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
            goal_x,
            track_half_width,
        )

        # Spaces
        observation_space = Box(low=-1e10, high=1e10, shape=(17,), dtype=np.float32)
        action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = observation_space
        self.action_space = action_space

        # Episode / task state
        self._reset_noise_scale = float(reset_noise_scale)
        self.max_steps = int(max_steps)
        self.goal_x = float(goal_x)
        self.goal_distance = self.goal_x + 0.5  # To prevent agent from stopping right before the line
        self.track_half_width = float(track_half_width)

        # PD gains
        self._kp = float(kp)
        self._kd = float(kd)

        # Counters
        self.steps = 0  # steps within current episode
        self.num_steps = 0  # total steps across episodes
        self.num_episodes = 0
        self.num_success = 0
        self.num_falls = 0

        # Flags & internals
        self.prev_x_pos = 0.0
        self.goal_reached = False
        self.line_crossed = False
        self.is_fallen = False
        self.prev_potential = None  # Will be initialized in reset_model

        # Init MuJoCo
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

    def step(self, action: np.ndarray):
        obs_now = self.get_obs()
        roll, pitch, yaw = self.get_euler_angles(obs_now)
        roll_vel, pitch_vel, yaw_vel = obs_now[12:15]

        a = np.asarray(action, dtype=np.float32)
        if a.size >= 2:
            wheel_tau, pitch_tau = float(a[0]), float(a[1])
        else:
            wheel_tau = float(a[0]) if a.size > 0 else 0.0
            pitch_tau = 0.0

        # PD torques for roll & yaw
        roll_tau = np.clip(-(self._kp * roll + self._kd * roll_vel) / 10.0, -1.0, 1.0)
        yaw_tau  = np.clip(-(self._kp * yaw  + self._kd * yaw_vel ) / 10.0, -1.0, 1.0)

        full_action = np.array([wheel_tau, roll_tau, pitch_tau, yaw_tau], dtype=np.float32)

        self.do_simulation(full_action, self.frame_skip)
        self.steps += 1
        self.num_steps += 1

        obs = self.get_obs()
        x_pos, y_pos = float(obs[0]), float(obs[1])
        euler = self.get_euler_angles(obs)  # (roll, pitch, yaw)

        if not self.is_fallen and self.check_fall(euler):
            self.num_falls += 1
            self.is_fallen = True

        reward, rcomps = self.calculate_reward(x_pos, y_pos, euler, obs)

        self.ep_return += reward

        self.update_success_count(x_pos)

        terminated = bool(
            self.is_fallen
            or abs(y_pos) > self.track_half_width
            or x_pos >= self.goal_x
        )
        truncated = bool(self.max_steps and self.steps >= self.max_steps)

        info = self.get_info(x_pos, y_pos, euler, reward, rcomps)

        if terminated or truncated:
            print(
                f"[iter {self.num_episodes}] "
                f"[steps {self.steps}] "
                f"[episodes {self.num_episodes}] "
                f"[reward {self._ep_return:.2f}] "
                f"[success count {self.num_success}] "
                f"[x distance {info.get('dist_to_goal', self.goal_distance - x_pos):.2f}]",
                flush=True
            )

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def reset_model(self):
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale

        qpos = (self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )).astype(np.float32)
        qvel = (self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )).astype(np.float32)

        # Normalize quaternion (wxyz)
        quat = qpos[3:7]
        qpos[3:7] = quat / (np.linalg.norm(quat) + 1e-8)

        self.set_state(qpos, qvel)

        # Episode state reset
        self.steps = 0
        self.prev_x_pos = float(qpos[0])
        self.goal_reached = False
        self.is_fallen = False
        self.line_crossed = False
        self.num_episodes += 1

        # Potential shaping init (based on actual starting position)
        x0 = float(qpos[0])
        self.prev_potential = -(self.goal_distance - x0)

        # For logging
        self.ep_return = 0.0

        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32, copy=False)

    def get_euler_angles(self, observation: np.ndarray) -> np.ndarray:
        quat_wxyz = observation[3:7]  # MuJoCo: (w, x, y, z)
        quat_xyzw = np.roll(quat_wxyz, -1)  # SciPy:  (x, y, z, w)
        return Rotation.from_quat(quat_xyzw).as_euler("xyz")

    def calculate_reward(
        self, x_pos: float, y_pos: float, euler: np.ndarray, obs: np.ndarray
    ) -> tuple[float, Dict[str, float]]:

        roll, pitch, yaw = euler

        pendulum_ang = float(obs[7])
        x_vel = float(obs[9])
        y_vel = float(obs[10])
        roll_vel  = float(obs[12])
        pitch_vel = float(obs[13])
        yaw_vel   = float(obs[14])
        pendulum_vel = float(obs[15])
        wheel_vel    = float(obs[16])

        # 1) Upright gate
        upright = np.exp(-4.0 * (roll**2 + pitch**2 + pendulum_ang**2))

        # 2) Potential-based shaping
        dist_to_goal = self.goal_distance - x_pos
        if dist_to_goal > 0.0:
            potential = -dist_to_goal
            shaping_reward = 4.0 * (potential - self.prev_potential) * upright
            self.prev_potential = potential
        else:
            shaping_reward = 0.0

        # 3) Speed-band bonus
        speed_reward = 0.3 * upright if (0.6 <= x_vel <= 1.2) else 0.0

        # 4) Lateral centering bonus (continuous)
        lat_bonus = 0.2 * np.exp(-2.0 * y_pos**2) * upright  # Bonus for staying near center
        
        # 5) Penalties
        tilt_pen  = -0.4  * (pitch**2)  - 0.05 * (roll**2)
        ang_pen = -0.02 * (pitch_vel**2) - 0.005 * (roll_vel**2)
        lat_vel_pen = -0.05 * abs(y_vel)  # Penalty for lateral velocity
        wheel_pen = -0.005 * (np.clip(wheel_vel, -15.0, 15.0) ** 2)
        pend_pen = (
            -0.3   * (np.clip(pendulum_ang, -15.0, 15.0) ** 2)
            -0.005 * (np.clip(pendulum_vel, -15.0, 15.0) ** 2)
        )

        # 6) Goal bonus
        goal_bonus = 0.0
        if (not self.line_crossed) and (x_pos >= self.goal_x):
            goal_bonus = 100.0
            self.line_crossed = True

        # 7) Final reward
        reward = shaping_reward + speed_reward + lat_bonus + goal_bonus \
                 + tilt_pen + ang_pen + lat_vel_pen + wheel_pen + pend_pen

        comps = {
            "upright": float(upright),
            "shaping": float(shaping_reward),
            "speed_reward": float(speed_reward),
            "lat_bonus": float(lat_bonus),
            "tilt_pen": float(tilt_pen),
            "ang_pen": float(ang_pen),
            "lat_vel_pen": float(lat_vel_pen),
            "wheel_pen": float(wheel_pen),
            "pendulum_pen": float(pend_pen),
            "goal_bonus": float(goal_bonus),
            "dist_to_goal": float(dist_to_goal),
        }
        return float(reward), comps

    def check_fall(self, euler: np.ndarray) -> bool:
        roll, pitch, _ = euler
        pendulum_ang = float(self.data.qpos[7])
        limit = np.pi / 3.0  # 60 deg
        return (abs(roll) > limit) or (abs(pitch) > limit) or (abs(pendulum_ang) > limit)

    def update_success_count(self, x_pos: float) -> None:
        if self.prev_x_pos < self.goal_x <= x_pos:
            self.num_success += 1
            self.goal_reached = True
        self.prev_x_pos = x_pos

    def get_info(
        self, x_pos: float, y_pos: float, euler: np.ndarray, reward: float, rcomps: Dict[str, float] | None = None
    ) -> Dict[str, Any]:
        roll, pitch, _ = euler
        info: Dict[str, Any] = {
            # Task state
            "x_position": x_pos,
            "y_position": y_pos,
            "goal_x": float(self.goal_x),
            "goal_distance": float(self.goal_distance),
            "goal_reached": self.goal_reached,

            # Attitude
            "unicycle_roll": roll,
            "unicycle_pitch": pitch,
            "pendulum_ang": float(self.data.qpos[7]),

            # Episode stats
            "steps": self.steps,
            "num_steps": self.num_steps,
            "num_success": self.num_success,
            "num_episodes": self.num_episodes,
            "num_falls": self.num_falls,
            "is_fallen": self.is_fallen,

            # Reward
            "reward": float(reward),
        }
        if rcomps:
            info.update(rcomps)
        return info