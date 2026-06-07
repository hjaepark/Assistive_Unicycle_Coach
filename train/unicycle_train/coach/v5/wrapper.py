# train/unicycle_train/coach/v5/wrapper.py

"""
V5 CoachWrapper

Identical to V4, plus trajectory-level latent z computation.

Manages two distinct phases:
  1. Coaching phase (K episodes): a_env = a_student + w_t * δ_t
  2. Eval phase (M episodes):     a_env = a_student (coach off)

V5 addition: during eval (self-practice) phase, wrapper accumulates
(state, action) pairs per episode. After phase ends, the LAST episode's
trajectory is encoded by the frozen autoencoder to produce z (32D).
z is appended to π_I observation.

Uses:
  - π_A (BC pretrained CoachActor, frozen, outputs δ_t per timestep)
  - encoder (frozen autoencoder encoder, outputs z from trajectory)
"""

import gymnasium as gym
import numpy as np
import torch


class CoachWrapper(gym.Wrapper):
    def __init__(self, env, assistance_policy, encoder, feature_mean, feature_std,
                 latent_dim=32, max_seq_len=1500, w_t=0):
        super().__init__(env)
        self.assistance_policy = assistance_policy
        self.encoder = encoder              # frozen autoencoder
        self.feature_mean = feature_mean    # (21,) numpy, for z-score normalization
        self.feature_std = feature_std      # (21,)
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.w_t = w_t

        # Phase mode: "coaching" or "eval"
        self._mode = "coaching"

        # Per-episode tracking
        self._last_obs = None
        self._last_reward = 0.0
        self._episode_reward = 0.0

        # Per-episode trajectory accumulator (for z computation in eval phase)
        self._ep_states = []       # list of state vectors (19D each)
        self._ep_actions = []      # list of action vectors (2D each)

        # Last completed self-practice episode trajectory (used for z)
        self._last_eval_states = None       # np.ndarray (T, 19) or None
        self._last_eval_actions = None      # np.ndarray (T, 2) or None

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
        if self._mode == "coaching" and self.w_t == 1:
            coach_obs = np.concatenate([
                self._last_obs,
                student_action,
                [self._last_reward],
            ]).astype(np.float32)

            delta_t = self.assistance_policy.predict(coach_obs, deterministic=False)
            exec_action = np.clip(student_action + delta_t, -1.0, 1.0)
            self._phase_coach_action_norms.append(float(np.linalg.norm(delta_t)))
        else:
            exec_action = student_action
            delta_t = np.zeros_like(student_action)

        # Track trajectory only in eval (self-practice) phase, for z computation
        if self._mode == "eval":
            self._ep_states.append(np.asarray(self._last_obs, dtype=np.float32))
            self._ep_actions.append(np.asarray(student_action, dtype=np.float32))

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

            # Save trajectory of just-finished eval episode (overwrites previous)
            if self._mode == "eval" and len(self._ep_actions) > 0:
                self._last_eval_states = np.stack(self._ep_states)    # (T, 19)
                self._last_eval_actions = np.stack(self._ep_actions)  # (T, 2)
            self._ep_states = []
            self._ep_actions = []

        info["coach_action"] = delta_t
        info["w_t"] = self.w_t
        info["coach_mode"] = self._mode

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_reward = 0.0
        self._episode_reward = 0.0
        # Drop any partial mid-episode trajectory
        self._ep_states = []
        self._ep_actions = []
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

    @torch.no_grad()
    def _compute_z(self):
        """Encode last self-practice episode trajectory to latent z (32D).
        Returns zeros if no completed eval episode yet."""
        if self._last_eval_states is None or self._last_eval_actions is None:
            return np.zeros(self.latent_dim, dtype=np.float32)

        # Build (T, 21) feature, truncate to max_seq_len
        feat = np.concatenate(
            [self._last_eval_states, self._last_eval_actions], axis=1
        ).astype(np.float32)
        if feat.shape[0] > self.max_seq_len:
            feat = feat[:self.max_seq_len]
        T = feat.shape[0]

        # Z-score normalize using dataset stats
        feat = (feat - self.feature_mean) / self.feature_std

        # Pad to max_seq_len, build mask
        padded = np.zeros((self.max_seq_len, 21), dtype=np.float32)
        padded[:T] = feat
        mask = np.zeros((self.max_seq_len,), dtype=bool)
        mask[:T] = True

        # Encode (add batch dim)
        x = torch.from_numpy(padded).unsqueeze(0)   # (1, T_max, 21)
        m = torch.from_numpy(mask).unsqueeze(0)     # (1, T_max)
        z = self.encoder.encode(x, m)               # (1, 32)
        return z.squeeze(0).cpu().numpy().astype(np.float32)

    def get_pi_observation(self):
        summary = self.get_phase_summary()
        z = self._compute_z()
        return np.concatenate([
            [summary["avg_reward"], summary["success_rate"]],
            z,
        ]).astype(np.float32)