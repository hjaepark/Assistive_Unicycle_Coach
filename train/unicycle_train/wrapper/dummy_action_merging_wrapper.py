# train/unicycle_train/wrapper/dummy_action_merging_wrapper.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DummyActionMergeWrapper(gym.Wrapper):
    """
    Coach action is always zero. Expands obs if include_coach_in_obs=True.
    final_action = student_action + alpha*(0 - student_action).
    """
    def __init__(self, env, alpha=0.5, include_coach_in_obs=False):
        super().__init__(env)
        self.alpha = alpha
        self.include_coach_in_obs = include_coach_in_obs

        self.raw_obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        if self.include_coach_in_obs:
            low = np.concatenate([self.env.observation_space.low,
                                  np.full((self.action_dim,), -1.0)])
            high = np.concatenate([self.env.observation_space.high,
                                   np.full((self.action_dim,), 1.0)])
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.observation_space = self.env.observation_space

        self.action_space = self.env.action_space
        self._last_obs_raw = None

    def set_alpha(self, new_alpha: float):
        self.alpha = new_alpha

    def reset(self, **kwargs):
        obs_raw, info = self.env.reset(**kwargs)
        self._last_obs_raw = obs_raw
        coach_action = np.zeros(self.action_dim, dtype=np.float32)

        if self.include_coach_in_obs:
            obs = np.concatenate([obs_raw, coach_action], axis=0)
        else:
            obs = obs_raw

        return obs, info

    def step(self, student_action):
        if self._last_obs_raw is None:
            raise RuntimeError("Must reset before step.")

        coach_action = np.zeros(self.action_dim, dtype=np.float32)
        coach_action_prime = self.alpha * (coach_action - student_action)
        final_action = student_action + coach_action_prime
        final_action = np.clip(final_action, -1.0, 1.0)

        next_obs_raw, reward, done, truncated, info = self.env.step(final_action)
        info["reward"] = reward
        info["student_action"] = student_action
        info["coach_action"] = coach_action
        info["coach_action_prime"] = coach_action_prime

        if self.include_coach_in_obs:
            next_coach_action = np.zeros(self.action_dim, dtype=np.float32)
            next_obs = np.concatenate([next_obs_raw, next_coach_action], axis=0)
        else:
            next_obs = next_obs_raw

        self._last_obs_raw = next_obs_raw
        return next_obs, reward, done, truncated, info
