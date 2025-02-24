# student_hf_env.py

import os
from collections import deque
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

class StudentHF(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }
    
    """
    ##########################################################################################################################
    ##########################################################################################################################
    ######################## Student agent + human factor environment code for the Unicycle Project ##########################
    ##########################################################################################################################
    ##########################################################################################################################

    StudentHF Environment (with Human Factors)
    ------------------------------------------
    - Includes action delay (action_delay steps)
    - Now includes 'action history' in observation
      so that the RL agent can see which actions 
      are in the pipeline (not yet applied) or 
      recently applied, to mitigate delay issues.

    """

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 2, # 0.005 timestep in XML * frame_skip 2 = 0.01sec in sim time
        default_camera_config: Dict[str, Union[float, int, np.ndarray]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        max_steps: int = 1000, 
        
        #Human factor
        safe_coef: float = 0.02,
        safe_margin: float = 0.3,
        sensor_noise_std: float = 0.0,  # Std dev for observation noise
        action_delay: int = 3,
        **kwargs,
    ):
        
        # Unicycle XML Path
        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "unicycle_pendulum_3d.xml")

        """
        # Initialize EzPickle
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, max_steps, **kwargs)
        """

        # Initialize EzPickle
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            reset_noise_scale,
            max_steps,
            safe_coef,
            safe_margin,
            sensor_noise_std,
            action_delay,
            **kwargs
        )

        # Original Obs dimension = 22
        # Add (action_delay * action_dim)= (3 * 3)= 9 for action_delay=3
        self._original_obs_dim = 22
        self.action_dim = 3
        self.action_delay = action_delay

        # Action space
        self.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Extended observation space
        aug_obs_dim = self._original_obs_dim + (self.action_dim * self.action_delay)
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(aug_obs_dim,),
            dtype=np.float64
        )

        # Store other parameters
        self._reset_noise_scale = reset_noise_scale
        self.max_steps = max_steps
        self.num_steps = 0 # Cumulative steps
        self.num_episodes = 0 # Cumulative episodes
        self.num_success = 0 # Cumulative success count
        self.prev_x_pos = 0
        self.target_success_count = 1000
        self.goal_distance = 12.0
        self.goal_reached = False

        self.safe_coef = safe_coef
        self.safe_margin = safe_margin
        self.sensor_noise_std = sensor_noise_std

        # Buffers for actual action delay
        # - action_buffer: used to find 'real_action' for do_simulation
        # - action_history: used to augment the observation
        from collections import deque
        self.action_buffer = deque(maxlen=self.action_delay)  # for delayed execution
        self.action_history = deque(maxlen=self.action_delay) # for observation augmentation

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
        """
        1) Decide the real action to apply in the physics engine 
           based on the action delay mechanism.
        2) do_simulation(real_action).
        3) Compute observation, reward, done, etc.
        """

        self.steps += 1 # Count to 1000 and reset
        self.num_steps += 1 # Global count

        # Action delay
        if self.action_delay > 0:
            if len(self.action_buffer) < self.action_delay:
                # Buffer isn't filled yet
                real_action = np.zeros_like(action)
            else:
                # Oldest action in the buffer
                real_action = self.action_buffer[0]

            # After deciding real_action, push new action (which will be used in future steps)
            self.action_buffer.append(action)

        else:
            # No delay
            real_action = action
        
        
        # 액션 히스토리에 현재 step의 action 기록
        self.action_history.append(action)

        # Perform simulation with the "real" action
        self.do_simulation(real_action, self.frame_skip)

        # Get current x & y position
        observation = self._get_obs()
        x_pos, y_pos = observation[:2]

        # Update success count
        self._update_success_count(x_pos)

        # Calculate euler angles for unicycle & pendulum
        unicycle_euler, pendulum_euler = self._get_euler_angles(observation)

        # Calculate reward
        reward = self._calculate_reward(x_pos, y_pos, unicycle_euler, pendulum_euler, observation)

        # Check termination conditions
        terminated = self._check_termination(x_pos, y_pos, unicycle_euler, pendulum_euler)
        truncated = bool(self.steps >= self.max_steps)

        # Collect additional information
        info = self._get_info(x_pos, y_pos, unicycle_euler, pendulum_euler, reward)

        # Render if in human mode
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        """
        Reset the environment state (qpos, qvel)
        and clear the action buffers.
        """
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=noise_low, high=noise_high)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=noise_low, high=noise_high)

        self.set_state(qpos, qvel)

        self.steps = 0
        self.prev_x_pos = 0
        self.goal_reached = False
        self.num_episodes += 1

        # Clear buffers
        self.action_buffer.clear()
        self.action_history.clear()

        # Initialize the action_history with zeros
        for _ in range(self.action_delay):
            self.action_history.append(np.zeros(self.action_dim, dtype=np.float32))

        obs = self._get_obs()
        return obs
    
    def _get_obs(self):
        """
        Returns:
            ndarray of shape (22 + action_delay*3,).
            The original 22-dim state plus
            the concatenated past `action_delay` actions.
        """
        # 1) base obs: 22-dim
        base_obs = np.concatenate([self.data.qpos, self.data.qvel]).ravel()

        # 2) flatten the stored actions (len=self.action_delay)
        past_acts = list(self.action_history)
        # If not filled up yet, the code in reset_model & step() ensures we have 
        # enough items. Alternatively, we could zero-pad here.

        past_acts_arr = np.concatenate(past_acts, axis=0)  # shape = (3 * action_delay,)

        # 3) combine
        obs = np.concatenate([base_obs, past_acts_arr], axis=0)

        # 4) sensor noise
        if self.sensor_noise_std > 0.0:
            noise = self.np_random.normal(loc=0.0, scale=self.sensor_noise_std, size=obs.shape)
            obs += noise

        return obs
    
    def _update_success_count(self, x_pos: float) -> None:
        # Count +1 if reached 12m destination
        if self.prev_x_pos < 12 and x_pos >= 12:
            self.num_success += 1
            self.goal_reached = True
        
        self.prev_x_pos = x_pos

    def _get_euler_angles(self, observation):
        # Convert quaternions to euler angles
        unicycle_quat = observation[3:7]
        pendulum_quat = observation[7:11]
        
        unicycle_euler = Rotation.from_quat(np.roll(unicycle_quat, -1)).as_euler('xyz')
        pendulum_euler = Rotation.from_quat(np.roll(pendulum_quat, -1)).as_euler('xyz')
        
        return unicycle_euler, pendulum_euler
    
    def _calculate_reward(self, x_pos, y_pos, unicycle_euler, pendulum_euler, observation):
        unicycle_roll, unicycle_pitch, _ = unicycle_euler
        pendulum_roll, pendulum_pitch, _ = pendulum_euler
        
        # Balance reward
        balance_reward = 0.5 * (1.0 - 0.5 * (unicycle_roll**2 + unicycle_pitch**2 + pendulum_roll**2 + pendulum_pitch**2))
        
        # Forward reward
        forward_reward = 2.0 * x_pos
        
        # Velocity reward
        velocity_reward = observation[12]  # x-velocity
        
        # Penalty for excessive tilt
        tilt_penalty = -5.0 if (abs(unicycle_roll) > np.pi/4 or abs(unicycle_pitch) > np.pi/4 or
                                abs(pendulum_roll) > np.pi/4 or abs(pendulum_pitch) > np.pi/4) else 0.0
        
        # Penalty for excessive wheel speed
        wheel_speed_penalty = -0.05 * observation[21]**2
        
        # Calculate total reward
        reward = balance_reward + forward_reward + velocity_reward + tilt_penalty + wheel_speed_penalty

        # Human Factor
        if self.safe_coef > 0.0:
            safe_bonus = 0.0
            safe_angles = [unicycle_roll, unicycle_pitch, pendulum_roll, pendulum_pitch]
            for alpha in safe_angles:
                abs_a = abs(alpha)
                if abs_a < self.safe_margin:
                    ratio = abs_a / self.safe_margin  # 0~1
                    safe_bonus += self.safe_coef * (1.0 + np.cos(np.pi * ratio)) / 2.0

            reward += safe_bonus
        
        # Additional reward for reaching goal
        if self.goal_reached:
            reward += 100
        
        return reward
    
    def _check_termination(self, x_pos, y_pos, unicycle_euler, pendulum_euler):
        unicycle_roll, unicycle_pitch, _ = unicycle_euler
        pendulum_roll, pendulum_pitch, _ = pendulum_euler
        
        # Check termination conditions
        return bool(
            # y-axis movement limit
            abs(y_pos) > 2.0 or

            # Excessive unicycle tilt
            abs(unicycle_roll) > np.pi/3 or
            abs(unicycle_pitch) > np.pi/3 or
            
            # Excessive pendulum tilt
            abs(pendulum_roll) > np.pi/3 or
            abs(pendulum_pitch) > np.pi/3 or

            self.goal_reached or

            # Required success count achieved
            self.num_success >= self.target_success_count
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
            "num_steps": self.num_steps,
            "num_success": self.num_success,
            "num_episodes": self.num_episodes,
            "reward": reward,
        }