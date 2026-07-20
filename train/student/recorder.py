# train/student/recorder.py

from pathlib import Path

import gymnasium as gym
import numpy as np


class Recorder(gym.Wrapper):
    def __init__(self, env, out_dir, extra_keys=()):
        super().__init__(env)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.extra_keys = tuple(extra_keys)
        self.ep_idx = 0
        self._reset_buffers()

    def _reset_buffers(self):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.extra_buf = {k: [] for k in self.extra_keys}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_buffers()
        self.obs_buf.append(np.asarray(obs))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.act_buf.append(np.asarray(action))
        self.rew_buf.append(float(reward))
        self.obs_buf.append(np.asarray(obs))
        for k in self.extra_keys:
            self.extra_buf[k].append(float(info.get(k, np.nan)))
        if terminated or truncated:
            self._dump(terminated, truncated, info)
            self.ep_idx += 1
        return obs, reward, terminated, truncated, info

    def _dump(self, terminated, truncated, info):
        payload = dict(
            observations=np.asarray(self.obs_buf),  # (T+1, obs_dim)
            actions=np.asarray(self.act_buf),       # (T, act_dim)
            rewards=np.asarray(self.rew_buf),       # (T,)
            terminated=bool(terminated),
            truncated=bool(truncated),
            goal_reached=bool(info.get("goal_reached", False)),
            is_fallen=bool(info.get("is_fallen", False)),
            episode=self.ep_idx,
        )
        for k in self.extra_keys:
            payload[f"extra_{k}"] = np.asarray(self.extra_buf[k])  # (T,)
        np.savez_compressed(self.out_dir / f"ep{self.ep_idx:06d}.npz", **payload)