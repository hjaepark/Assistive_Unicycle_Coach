# train/unicycle_train/wrapper/action_merging_wrapper.py

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from gymnasium import spaces

class ActionMergeWrapper(gym.Wrapper):
    """
    Merges student action with a pre-trained coach action:
    final_action = student + alpha*(coach - student).
    Obs includes [raw_obs, coach_action].
    """
    def __init__(self, env, coach_model_path, alpha=0.0):
        super().__init__(env)
        self.coach_model = SAC.load(coach_model_path, env=None)
        self.alpha = alpha

        self.raw_obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        low = np.concatenate([self.env.observation_space.low,
                              np.full((self.action_dim,), -1.0)])
        high = np.concatenate([self.env.observation_space.high,
                               np.full((self.action_dim,), 1.0)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = self.env.action_space
        self._last_obs_raw = None

    def set_alpha(self, new_alpha: float):
        self.alpha = new_alpha

    def reset(self, **kwargs):
        obs_raw, info = self.env.reset(**kwargs)
        self._last_obs_raw = obs_raw
        coach_action, _ = self.coach_model.predict(obs_raw, deterministic=True)
        obs = np.concatenate([obs_raw, coach_action], axis=0)
        return obs, info

    def step(self, student_action):
        if self._last_obs_raw is None:
            raise RuntimeError("Must reset before step.")

        coach_action, _ = self.coach_model.predict(self._last_obs_raw, deterministic=True)
        coach_action_prime = self.alpha * (coach_action - student_action)
        final_action = student_action + coach_action_prime
        final_action = np.clip(final_action, -1.0, 1.0)

        next_obs_raw, reward, done, truncated, info = self.env.step(final_action)
        info["reward"] = reward
        info["student_action"] = student_action
        info["coach_action"] = coach_action
        info["coach_action_prime"] = coach_action_prime

        next_coach_action, _ = self.coach_model.predict(next_obs_raw, deterministic=True)
        next_obs = np.concatenate([next_obs_raw, next_coach_action], axis=0)

        self._last_obs_raw = next_obs_raw
        return next_obs, reward, done, truncated, info