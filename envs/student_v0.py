import os
from typing import Dict, Union

import numpy as np
from scipy.spatial.transform import Rotation

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# Default camera configuration
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
}

class Student(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }
    
    """
    ##########################################################################################################################
    ##########################################################################################################################
    ############################ Student agent environment code for the Unicycle Project #####################################
    ##########################################################################################################################
    ##########################################################################################################################


    ## Description
    This environment simulates a unicycle with an inverted pendulum attached to it. 
    The goal is to balance both the unicycle and the pendulum while moving the unicycle forward for 12 meters along the x-axis.

    ## Action Space
    The agent takes a 3-element vector for actions.
    | Num | Action                           | Control Min | Control Max | Name (in XML file) | Joint        | Unit         |
    |-----|----------------------------------|-------------|-------------|--------------------|--------------|--------------|
    | 0   | Torque applied on the wheel      | -1          | 1           | wheel_motor        | wheel_joint  | torque (N m) |
    | 1   | Roll stabilization torque        | -1          | 1           | roll_stabilizer    | free_joint   | torque (N m) |
    | 2   | Yaw control torque               | -1          | 1           | yaw_control        | free_joint   | torque (N m) |
    
    Solution model will take action based on the environment after student's action taken

    ## Observation Space
    The observation consists of:
    | Num | Observation                                         | Min  | Max | Name (in corresponding XML file) | Joint |        Type (Unit)          |
    |-----|-----------------------------------------------------|------|-----|----------------------------------|-------|-----------------------------|
    | 0   | position of the unicycle in the x-direction          | -Inf | Inf | unicycle                         | free  | position (m)               |
    | 1   | position of the unicycle in the y-direction          | -Inf | Inf | unicycle                         | free  | position (m)               |
    | 2   | position of the unicycle in the z-direction          | -Inf | Inf | unicycle                         | free  | position (m)               |
    | 3   | w-component of the unicycle's quaternion             | -1   | 1   | unicycle (quaternion)            | free  | quaternion component       |
    | 4   | x-component of the unicycle's quaternion             | -1   | 1   | unicycle (quaternion)            | free  | quaternion component       |
    | 5   | y-component of the unicycle's quaternion             | -1   | 1   | unicycle (quaternion)            | free  | quaternion component       |
    | 6   | z-component of the unicycle's quaternion             | -1   | 1   | unicycle (quaternion)            | free  | quaternion component       |
    | 7   | w-component of the pendulum's quaternion             | -1   | 1   | pendulum (quaternion)            | ball  | quaternion component       |
    | 8   | x-component of the pendulum's quaternion             | -1   | 1   | pendulum (quaternion)            | ball  | quaternion component       |
    | 9   | y-component of the pendulum's quaternion             | -1   | 1   | pendulum (quaternion)            | ball  | quaternion component       |
    | 10  | z-component of the pendulum's quaternion             | -1   | 1   | pendulum (quaternion)            | ball  | quaternion component       |
    | 11  | angle of the wheel                                   | -Inf | Inf | wheel                            | hinge | angle (rad)                |
    | 12  | velocity of the unicycle in the x-direction          | -Inf | Inf | unicycle                         | free  | velocity (m/s)             |
    | 13  | velocity of the unicycle in the y-direction          | -Inf | Inf | unicycle                         | free  | velocity (m/s)             |
    | 14  | velocity of the unicycle in the z-direction          | -Inf | Inf | unicycle                         | free  | velocity (m/s)             |
    | 15  | angular velocity of the unicycle about x-axis        | -Inf | Inf | unicycle                         | free  | angular velocity (rad/s)   |
    | 16  | angular velocity of the unicycle about y-axis        | -Inf | Inf | unicycle                         | free  | angular velocity (rad/s)   |
    | 17  | angular velocity of the unicycle about z-axis        | -Inf | Inf | unicycle                         | free  | angular velocity (rad/s)   |
    | 18  | angular velocity of the pendulum about x-axis        | -Inf | Inf | pendulum                         | ball  | angular velocity (rad/s)   |
    | 19  | angular velocity of the pendulum about y-axis        | -Inf | Inf | pendulum                         | ball  | angular velocity (rad/s)   |
    | 20  | angular velocity of the pendulum about z-axis        | -Inf | Inf | pendulum                         | ball  | angular velocity (rad/s)   |
    | 21  | angular velocity of the wheel                        | -Inf | Inf | wheel                            | hinge | angular velocity (rad/s)   |


    The observation is a `ndarray` with shape `(22,)` where the elements correspond to the table above.

    ## Rewards
    The reward function is designed to encourage the unicycle to move forward while maintaining balance:
    - A positive reward for moving forward along the x-axis
    - A penalty for tilting (both the unicycle and the pendulum)
    - A penalty for excessive wheel speed
    - A large bonus for reaching the 12-meter goal
    Solution model doesn't have separated rewards

    ## Starting State
    The unicycle starts near the origin with a slightly random position and orientation.

    ## Episode Termination
    The episode ends when:
    1. The unicycle falls over (excessive tilt)
    2. The pendulum falls over (excessive tilt)
    3. The unicycle reaches the 12-meter goal
    4. The maximum number of steps is reached (1000 steps)

    ## Solved Requirement
    The environment is considered solved when the agent can consistently move the unicycle 12 meters forward while maintaining balance.
    To obtain the ideal reference model, training will be end once the unicycle agent passed 12 meters 10 times. (verified through testing 10^x goal reaching experiment)
    """

    # Environment Constants
    GOAL_DISTANCE = 12.0
    MAX_STEPS = 1000
    REQUIRED_SUCCESSES = 10
    
    # Reward Constants
    BALANCE_SCALE = 0.5
    FORWARD_SCALE = 2.0
    TILT_PENALTY = -5.0
    WHEEL_SPEED_PENALTY_SCALE = -0.05
    Y_PENALTY_SCALE = -0.5
    GOAL_REWARD = 100.0
    
    # Angle Thresholds (in radians)
    MAX_TILT_ANGLE = np.pi/3
    PENALTY_TILT_ANGLE = np.pi/4

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int, np.ndarray]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        **kwargs,
    ):
        # Use default XML file if not provided
        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "unicycle_pendulum_3d.xml")
        
        # Initialize EzPickle
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, self.MAX_STEPS, **kwargs)
        
        # Define observation and action spaces
        observation_space = Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float64)
        self.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Set environment parameters
        self._reset_noise_scale = reset_noise_scale
        self.max_steps = self.MAX_STEPS
        self.total_episodes = 0
        self.success_count = 0
        self.prev_x = 0
        self.goal_reached = False

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
        
        # Get current observation
        observation = self._get_obs()
        x_pos, y_pos = observation[:2]
        
        # Update success count
        self._update_success_count(x_pos)
        
        # Calculate euler angles for unicycle and pendulum
        unicycle_euler, pendulum_euler = self._get_euler_angles(observation)
        
        # Compute reward
        reward = self._compute_reward(x_pos, y_pos, unicycle_euler, pendulum_euler, observation)
        
        # Check termination conditions
        terminated = self._check_termination(x_pos, y_pos, unicycle_euler, pendulum_euler)
        truncated = bool(self.steps >= self.max_steps)
        
        # Collect additional information
        info = self._get_info(x_pos, y_pos, unicycle_euler, pendulum_euler, reward)
        
        # Render if in human mode
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info

    def _update_success_count(self, x_pos: float) -> None:
        # Increase success count when first crossing goal distance mark
        if self.prev_x < self.GOAL_DISTANCE and x_pos >= self.GOAL_DISTANCE:
            self.success_count += 1
            self.goal_reached = True
        
        self.prev_x = x_pos

    def _get_euler_angles(self, observation):
        # Convert quaternions to euler angles
        unicycle_quat = observation[3:7]
        pendulum_quat = observation[7:11]
        
        unicycle_euler = Rotation.from_quat(np.roll(unicycle_quat, -1)).as_euler('xyz')
        pendulum_euler = Rotation.from_quat(np.roll(pendulum_quat, -1)).as_euler('xyz')
        
        return unicycle_euler, pendulum_euler

    def _compute_reward(self, x_pos, y_pos, unicycle_euler, pendulum_euler, observation):
        unicycle_roll, unicycle_pitch, _ = unicycle_euler
        pendulum_roll, pendulum_pitch, _ = pendulum_euler
        
        # Normalize angles to [-1, 1] range for smoother rewards
        balance_term = (unicycle_roll**2 + unicycle_pitch**2 + 
                    pendulum_roll**2 + pendulum_pitch**2) / (self.PENALTY_TILT_ANGLE**2)
        
        # Compute individual reward components
        balance_reward = self.BALANCE_SCALE * (1.0 - 0.5 * balance_term)
        forward_reward = self.FORWARD_SCALE * x_pos
        velocity_reward = max(min(observation[12], 2.0), -1.0)  # Clipped x-velocity reward
        
        # Penalties
        tilt_penalty = self.TILT_PENALTY if (abs(unicycle_roll) > self.PENALTY_TILT_ANGLE or 
                                        abs(unicycle_pitch) > self.PENALTY_TILT_ANGLE or
                                        abs(pendulum_roll) > self.PENALTY_TILT_ANGLE or 
                                        abs(pendulum_pitch) > self.PENALTY_TILT_ANGLE) else 0.0
        
        wheel_speed_penalty = self.WHEEL_SPEED_PENALTY_SCALE * min(observation[21]**2, 100)
        y_penalty = self.Y_PENALTY_SCALE * min(abs(y_pos), 2.0)
        
        # Combine rewards
        reward = (balance_reward + forward_reward + velocity_reward + 
                tilt_penalty + wheel_speed_penalty + y_penalty)
        
        # Goal reward
        if self.goal_reached:
            reward += self.GOAL_REWARD
        
        return reward

    def _check_termination(self, x_pos, y_pos, unicycle_euler, pendulum_euler):
        unicycle_roll, unicycle_pitch, _ = unicycle_euler
        pendulum_roll, pendulum_pitch, _ = pendulum_euler
        
        # Check termination conditions
        return bool(
            abs(unicycle_roll) > self.MAX_TILT_ANGLE or  # Excessive unicycle tilt
            abs(unicycle_pitch) > self.MAX_TILT_ANGLE or
            abs(pendulum_roll) > self.MAX_TILT_ANGLE or  # Excessive pendulum tilt
            abs(pendulum_pitch) > self.MAX_TILT_ANGLE or
            self.goal_reached or  # Goal reached
            self.success_count >= self.REQUIRED_SUCCESSES  # Required success count achieved
        )

    def _get_info(self, x_pos, y_pos, unicycle_euler, pendulum_euler, reward):
        # Collect additional information
        unicycle_roll, unicycle_pitch, _ = unicycle_euler
        pendulum_roll, pendulum_pitch, _ = pendulum_euler
        
        return {
            "unicycle_roll": unicycle_roll,
            "unicycle_pitch": unicycle_pitch,
            "pendulum_roll": pendulum_roll,
            "pendulum_pitch": pendulum_pitch,
            "x_position": x_pos,
            "y_position": y_pos,
            "goal_reached": self.goal_reached,
            "steps": self.steps,
            "success_count": self.success_count,
            "total_episodes": self.total_episodes,
            "reward": reward,
        }
    
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # More controlled initial state
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        # Add small noise to position (except height)
        qpos[0:2] += self.np_random.uniform(low=noise_low, high=noise_high, size=2)
        
        # Add small noise to orientation
        qpos[3:7] += self.np_random.uniform(low=noise_low/2, high=noise_high/2, size=4)
        qpos[3:7] /= np.linalg.norm(qpos[3:7])  # Normalize quaternion
        
        # Add small noise to velocities
        qvel += self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        
        self.set_state(qpos, qvel)
        
        # Reset episode variables
        self.steps = 0
        self.prev_x = 0
        self.goal_reached = False
        self.total_episodes += 1
        
        return self._get_obs()

    def _get_obs(self):
        # Get current observation (position and velocity information)
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()