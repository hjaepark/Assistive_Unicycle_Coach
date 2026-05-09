# train/unicycle_train/coach/v4/wrapper.py

"""
V4 CoachWrapper — identical to V3 CoachWrapper.
Each worker process has its own wrapper (one student per worker).

Manages two distinct phases:
  1. Coaching phase (K episodes): a_env = a_student + w_t * δ_t
  2. Eval phase (M episodes):     a_env = a_student (coach off)

Uses:
  - π_I (external, decides w_t per coaching phase, NOT inside wrapper)
  - π_A (BC pretrained CoachActor, frozen, outputs δ_t per timestep)
"""

import gymnasium as gym
import numpy as np


class CoachWrapper(gym.Wrapper):
    def __init__(self, env, assistance_policy, w_t=0):
        super().__init__(env)
        self.assistance_policy = assistance_policy
        self.w_t = w_t

        # Phase mode: "coaching" or "eval"
        self._mode = "coaching"

        # Per-episode tracking
        self._last_obs = None
        self._last_reward = 0.0
        self._episode_reward = 0.0

        # Phase-level tracking (accumulates across episodes within a phase)
        self._phase_episode_rewards = []
        self._phase_episode_successes = []
        self._phase_coach_action_norms = []

    def set_mode(self, mode):
        assert mode in ("coaching", "eval"), f"Invalid mode: {mode}"
        self._mode = mode
        self._phase_episode_rewards = []
        self._phase_episode_successes = []
        self._phase_coach_action_norms = []

    def set_intervention(self, w_t):
        self.w_t = w_t

    def step(self, student_action):
        # In coaching phase, scale delta_t by w_t before adding to student action.
        # w_t is now a float in [-1, 1] rather than binary {0, 1}.
        # Examples:
        #   w_t =  0.0  -> exec_action = student_action (no coach effect)
        #   w_t =  0.5  -> exec_action = student_action + 0.5 * delta_t (partial assist)
        #   w_t =  1.0  -> exec_action = student_action + delta_t (full assist, V4 behavior)
        #   w_t = -0.5  -> exec_action = student_action - 0.5 * delta_t (partial resist, V4.2)
        #   w_t = -1.0  -> exec_action = student_action - delta_t (full resist, V4.2)
        #
        # The 1e-8 threshold avoids unnecessary pi_A forward pass when w_t is effectively zero.
        if self._mode == "coaching" and abs(self.w_t) > 1e-8:
            coach_obs = np.concatenate([
                self._last_obs,
                student_action,
                [self._last_reward],
            ]).astype(np.float32)

            # pi_A outputs the full assistive correction delta_t (unchanged from V4)
            delta_t = self.assistance_policy.predict(coach_obs, deterministic=False)

            # KEY V4.1 CHANGE: scale delta_t by w_t before adding
            # This is the only mathematical difference from V4
            scaled_delta = self.w_t * delta_t
            exec_action = np.clip(student_action + scaled_delta, -1.0, 1.0)

            # Log the scaled (effective) coach action norm, not the raw delta norm.
            # This metric now reflects what was actually applied to the environment,
            # so a coach choosing w_t=0.25 will show smaller norms than one choosing w_t=1.0.
            self._phase_coach_action_norms.append(float(np.linalg.norm(scaled_delta)))
        else:
            exec_action = student_action
            delta_t = np.zeros_like(student_action)

        # Rest of the function is identical to V4
        obs, reward, terminated, truncated, info = self.env.step(exec_action)
        self._episode_reward += reward
        self._last_obs = obs
        self._last_reward = reward

        done = terminated or truncated
        if done:
            self._phase_episode_rewards.append(self._episode_reward)
            self._phase_episode_successes.append(
                1.0 if info.get("goal_reached", False) else 0.0
            )
            self._episode_reward = 0.0

        info["coach_action"] = delta_t
        info["w_t"] = self.w_t
        info["coach_mode"] = self._mode

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_reward = 0.0
        self._episode_reward = 0.0
        return obs, info

    def get_phase_summary(self):
        n_episodes = len(self._phase_episode_rewards)

        if n_episodes == 0:
            return {
                "avg_reward": 0.0,
                "success_rate": 0.0,
                "n_episodes": 0,
                "avg_coach_action_norm": 0.0,
            }

        return {
            "avg_reward": float(np.mean(self._phase_episode_rewards)),
            "success_rate": float(np.mean(self._phase_episode_successes)),
            "n_episodes": n_episodes,
            "avg_coach_action_norm": (
                float(np.mean(self._phase_coach_action_norms))
                if self._phase_coach_action_norms
                else 0.0
            ),
        }

    def get_pi_observation(self):
        summary = self.get_phase_summary()
        return np.array([
            summary["avg_reward"],
            summary["success_rate"],
        ], dtype=np.float32)