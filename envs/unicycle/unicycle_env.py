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
The agent takes a 4-element vector for actions.

| Num | Action                                 | Control Range | Name (in XML)           | Joint         | Gear Factor | Effective Torque (N·m)   |
|-----|----------------------------------------|---------------|-------------------------|---------------|------------ |--------------------------|
| 0   | Torque applied on the wheel            | -1 to 1       | wheel_motor             | wheel_joint   | 25          | -25 to 25                |
| 1   | Roll control torque (about x-axis)     | -1 to 1       | roll_control_motor      | free_joint    | 10          | -10 to 10                |
| 2   | Pitch control torque (about y-axis)    | -1 to 1       | pitch_control_motor     | free_joint    | 10          | -10 to 10                |
| 3   | Yaw control torque  (about z-axis)     | -1 to 1       | yaw_control_motor       | free_joint    | 10          | -10 to 10                |

Solution model will take action based on the environment after student's action taken.
"""

###################################################
## Observation Space
###################################################
"""
The observation consists of:

| Num | Observation                          | Min   | Max   | Type (Unit)                |
|-----|--------------------------------------|-------|-------|----------------------------|
| 0   | position in the x-direction          | -Inf  | Inf   | position (m)               |
| 1   | position in the y-direction          | -Inf  | Inf   | position (m)               |
| 2   | position in the z-direction          | -Inf  | Inf   | position (m)               |
| 3   | w-component of the unicycle's orientation | -1    | 1     | quaternion component       |
| 4   | x-component of the unicycle's orientation | -1    | 1     | quaternion component       |
| 5   | y-component of the unicycle's orientation | -1    | 1     | quaternion component       |
| 6   | z-component of the unicycle's orientation | -1    | 1     | quaternion component       |
| 7   | angle of the wheel                   | -Inf  | Inf   | angle (rad)                |
| 8   | velocity in the x-direction          | -Inf  | Inf   | velocity (m/s)             |
| 9   | velocity in the y-direction          | -Inf  | Inf   | velocity (m/s)             |
| 10  | velocity in the z-direction          | -Inf  | Inf   | velocity (m/s)             |
| 11  | angular velocity about x-axis        | -Inf  | Inf   | angular velocity (rad/s)   |
| 12  | angular velocity about y-axis        | -Inf  | Inf   | angular velocity (rad/s)   |
| 13  | angular velocity about z-axis        | -Inf  | Inf   | angular velocity (rad/s)   |
| 14  | angular velocity of the wheel        | -Inf  | Inf   | angular velocity (rad/s)   |

The observation is a `ndarray` with shape `(15,)` where the elements correspond to the table above.
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
    }

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 1,  # 0.005 timestep in XML * frame_skip 1 = 0.005sec in sim time
        default_camera_config: Dict[str, Union[float, int, np.ndarray]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        max_steps: int = 1500,
        **kwargs,
    ):
        # Unicycle XML Path
        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "unicycle_3d.xml")

        # Initialize EzPickle
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, max_steps, **kwargs)

        # Define observation and action spaces
        observation_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)
        self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Environment variables
        self._reset_noise_scale = reset_noise_scale
        self.max_steps = max_steps
        self.num_steps = 0  # Cumulative steps
        self.num_episodes = 0  # Cumulative episodes
        self.num_success = 0  # Cumulative success count
        self.prev_x_pos = 0
        self.target_success_count = 1000
        self.goal_distance = 5.5
        self.success_distance = 5.0  # New: distance to count as success
        self.goal_reached = False

        # Add fall counter
        self.num_falls = 0
        self._is_fallen = False

        # Initialize MujocoEnv
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # Set rendering FPS
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

    def step(self, action):
        # Perform simulation
        self.do_simulation(action, self.frame_skip)
        self.steps += 1
        self.num_steps += 1

        # Get current x & y position
        observation = self._get_obs()
        x_pos, y_pos = observation[:2]

        # Update success count
        self._update_success_count(x_pos)

        # Calculate Euler angles for unicycle
        unicycle_euler = self._get_euler_angles(observation)

        # Check if unicycle has fallen
        if not self._is_fallen and self._check_fall(unicycle_euler):
            self.num_falls += 1
            self._is_fallen = True

        # Calculate reward
        reward = self._calculate_reward(x_pos, y_pos, unicycle_euler, observation)

        # Check termination conditions
        terminated = self._check_termination(x_pos, y_pos, unicycle_euler)
        truncated = bool(self.steps >= self.max_steps)

        # Collect additional information
        info = self._get_info(x_pos, y_pos, unicycle_euler, reward)

        # Render if in human mode
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        # Reset model
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # Add noise to initial position and velocity
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )
        self.set_state(qpos, qvel)

        # Reset episode-related variables
        self.steps = 0
        self.prev_x_pos = 0
        self.goal_reached = False
        self.num_episodes += 1

        # Reset fall state
        self._is_fallen = False

        return self._get_obs()

    def _get_obs(self):
        # Get current observation (position + velocity)
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def _get_euler_angles(self, observation):
        # Convert quaternions to Euler angles
        unicycle_quat = observation[3:7]

        unicycle_euler = Rotation.from_quat(np.roll(unicycle_quat, -1)).as_euler('xyz')

        return unicycle_euler

    def _calculate_reward(self, x_pos, y_pos, unicycle_euler, observation):
        roll, pitch, yaw = unicycle_euler
        x_vel, y_vel, z_vel = observation[8], observation[9], observation[10]
        z_ang_vel = observation[13]
        wheel_ang_vel = observation[14]

        # (1) Balance Reward
        balance_reward = 1.0 - 0.5 * (roll**2 + pitch**2)
        balance_reward = np.clip(balance_reward, -1.0, 1.0)

        # (2) Forward Reward
        forward_reward = 0.5 * x_pos

        # (3) Forward Velocity Reward
        forward_vel_reward = 1.0 * x_vel

        # (4) Heading Alignment Reward
        yaw_error = abs(yaw)
        yaw_alignment_reward = np.exp(-2.0 * yaw_error)

        # (5) Lateral Velocity Penalty
        lateral_vel_penalty = -0.3 * abs(y_vel)
        
        # (6) Yaw Velocity Penalty
        yaw_velocity_penalty = -0.3 * (z_ang_vel**2)

        # (7) Tilt Penalty
        tilt_penalty = 0.0
        if abs(roll) > np.pi/3 or abs(pitch) > np.pi/3:
            tilt_penalty = -5.0

        # (8) Wheel Speed Penalty
        wheel_speed_penalty = -0.01 * (wheel_ang_vel**2)

        reward = (
            balance_reward
            + forward_reward
            + forward_vel_reward
            + yaw_alignment_reward
            + lateral_vel_penalty
            + yaw_velocity_penalty
            + tilt_penalty
            + wheel_speed_penalty
        )

        # Goal Reach Bonus
        if self.goal_reached:
            reward += 100.0

        return reward

    def _check_termination(self, x_pos, y_pos, unicycle_euler):
        unicycle_roll, unicycle_pitch, _ = unicycle_euler

        # Check termination conditions
        return bool(
            # y-axis movement limit
            abs(y_pos) > 4.0
            or
            # Reached final goal distance
            x_pos >= self.goal_distance
            or
            # Required success count achieved
            self.num_success >= self.target_success_count
            or
            # Fall condition
            self._check_fall(unicycle_euler)
        )

    def _check_fall(self, unicycle_euler):
        unicycle_roll, unicycle_pitch, _ = unicycle_euler
        return bool(
            abs(unicycle_roll) > np.pi * 4 / 9  # 80 degrees
            or abs(unicycle_pitch) > np.pi * 4 / 9  # 80 degrees
        )

    def _update_success_count(self, x_pos: float) -> None:
        # Count +1 if reached success distance (5m)
        if self.prev_x_pos < self.success_distance and x_pos >= self.success_distance:
            self.num_success += 1
            self.goal_reached = True
        self.prev_x_pos = x_pos

    def _get_info(self, x_pos, y_pos, unicycle_euler, reward):
        # Collect additional information
        unicycle_roll, unicycle_pitch, _ = unicycle_euler

        return {
            "unicycle_roll": unicycle_roll,
            "unicycle_pitch": unicycle_pitch,
            "x_position": x_pos,
            "y_position": y_pos,
            "goal_reached": self.goal_reached,
            "steps": self.steps,
            "num_steps": self.num_steps,
            "num_success": self.num_success,
            "num_episodes": self.num_episodes,
            "num_falls": self.num_falls,
            "is_fallen": self._is_fallen,
            "reward": reward,
        }
