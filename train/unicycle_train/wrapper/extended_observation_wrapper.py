from gymnasium.core import Wrapper
from gymnasium import spaces
import numpy as np

class ExtendedObservationWrapper(Wrapper):
    """
    Wrapper that extends the observation space with dummy action values.
    This is used to maintain a consistent observation space structure
    even when the actual coach actions are not available.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_dim = env.action_space.shape[0]
        
        # Create extended observation space
        low = np.concatenate([env.observation_space.low,
                            np.full((self.action_dim,), -1.0)])
        high = np.concatenate([env.observation_space.high,
                             np.full((self.action_dim,), 1.0)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Extend observation with zeros (dummy action)
        extended_obs = np.concatenate([obs, np.zeros(self.action_dim)])
        return extended_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # Remove return_info from kwargs to avoid the error
        obs = self.env.reset(**kwargs)
        # Extend observation with zeros (dummy action)
        extended_obs = np.concatenate([obs, np.zeros(self.action_dim)])
        return extended_obs 