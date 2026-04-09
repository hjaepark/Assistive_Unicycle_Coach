# train/unicycle_train/coach/v3/wrapper.py

"""
V3 CoachWrapper — Selective Intervention Coaching

Manages two distinct phases:
  1. Coaching phase (K episodes): a_env = a_student + w_t * δ_t
  2. Eval phase (M episodes):     a_env = a_student (coach off, no student update)

Combines two coach policies:
  - π_I (InterventionPolicy): decides w_t ∈ {0, 1} per coaching phase
  - π_A (BC pretrained CoachActor): outputs δ_t per timestep (frozen)

Tracks student progress (avg reward, success rate) across episodes
to provide π_I observation for next round's decision.
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
        # Reset phase tracking when mode changes
        self._phase_episode_rewards = []
        self._phase_episode_successes = []
        self._phase_coach_action_norms = []

    def set_intervention(self, w_t):
        self.w_t = w_t

    def step(self, student_action):
        if self._mode == "coaching" and self.w_t == 1:
            coach_obs = np.concatenate([
                self._last_obs,
                student_action,
                [self._last_reward],
            ]).astype(np.float32)

            # Get δ_t from frozen assistance policy
            delta_t = self.assistance_policy.predict(coach_obs, deterministic=False)

            exec_action = np.clip(student_action + delta_t, -1.0, 1.0)
            self._phase_coach_action_norms.append(float(np.linalg.norm(delta_t)))
        else:
            exec_action = student_action
            delta_t = np.zeros_like(student_action)

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