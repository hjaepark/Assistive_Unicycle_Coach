# train/unicycle_train/coach/test/mini_wrapper.py
"""
Minimal CoachWrapper for buffer-mislabeling diagnosis.

NO learned intervention policy (pi_I): intervention is a FIXED rule here,
governed by `intervene_rate` (probability of intervening each step).
NO autoencoder, NO z, NO phases, NO pi_observation.

pi_A (the frozen BC correction policy) IS used when an intervention fires,
because that is what "the coach helps" physically means.

On every step's `info`:
  - info["w_t"]:        1 if intervened this step, else 0
  - info["exec_action"]: action actually applied to the environment
so a custom replay buffer can relabel or skip intervened transitions.
"""

import gymnasium as gym
import numpy as np


class MiniCoachWrapper(gym.Wrapper):
    def __init__(self, env, assistance_policy, intervene_rate=1.0, rng_seed=0):
        super().__init__(env)
        self.assistance_policy = assistance_policy   # frozen pi_A, or None
        self.intervene_rate = float(intervene_rate)  # 1.0=always, 0.0=never
        self._rng = np.random.default_rng(rng_seed)
        self._last_obs = None
        self._last_reward = 0.0

    def set_intervene_rate(self, rate):
        self.intervene_rate = float(rate)

    def step(self, student_action):
        # Fixed-rule intervention: coin flip at intervene_rate
        fire = (self.assistance_policy is not None
                and self._rng.random() < self.intervene_rate)

        if fire:
            # pi_A observes [state, student_action, last_reward] (22D, matches BC)
            coach_obs = np.concatenate([
                self._last_obs, student_action, [self._last_reward],
            ]).astype(np.float32)
            delta_t = self.assistance_policy.predict(coach_obs, deterministic=False)
            exec_action = np.clip(student_action + delta_t, -1.0, 1.0)
            w_t = 1
        else:
            exec_action = student_action
            w_t = 0

        obs, reward, terminated, truncated, info = self.env.step(exec_action)
        self._last_obs = obs
        self._last_reward = reward

        info["w_t"] = w_t
        info["exec_action"] = np.asarray(exec_action, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_reward = 0.0
        return obs, info