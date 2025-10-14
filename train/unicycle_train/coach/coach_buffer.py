# train/unicycle_train/coach/coach_buffer.py

"""
Coach also uses off-policy algorithm (SAC), so we need to store its experiences,
and sample from them for training.
Stable-Baselines3 only provides ReplayBuffer for single agent, 
so it is necessary to implement custom buffer for the coach.
"""

import numpy as np

class CoachReplayBuffer:
    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.obs_dim = 22  # 19 + 1 + 2 ← Changed from 20
        self.action_dim = 2
        self.buffer = {
            "obs": np.zeros((capacity, self.obs_dim), dtype=np.float32),
            "action": np.zeros((capacity, self.action_dim), dtype=np.float32),
            "reward": np.zeros(capacity, dtype=np.float32),
            "next_obs": np.zeros((capacity, self.obs_dim), dtype=np.float32),
            "done": np.zeros(capacity, dtype=bool),
            "weight_assist": np.zeros(capacity, dtype=np.float32),
        }
        self.ptr = 0
        self.size = 0
    
    def add(self, obs, action, reward, next_obs, done, weight_assist):
        assert obs.shape == (self.obs_dim,), f"obs {obs.shape} != {(self.obs_dim,)}"
        assert next_obs.shape == (self.obs_dim,), f"next_obs {next_obs.shape} != {(self.obs_dim,)}"
        assert action.shape == (self.action_dim,), f"action {action.shape} != {(self.action_dim,)}"
        assert isinstance(done, (bool, np.bool_)), f"done must be bool, got {type(done)}"

        weight_assist = float(np.clip(weight_assist, 0.0, 1.0))

        i = self.ptr
        self.buffer["obs"][i] = obs
        self.buffer["action"][i] = action
        self.buffer["reward"][i] = reward
        self.buffer["next_obs"][i] = next_obs
        self.buffer["done"][i] = done
        self.buffer["weight_assist"][i] = weight_assist
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size=256):
        if self.size == 0:
            return {"size": 0, "fill_ratio": 0.0} # or raise ValueError("Buffer is empty")

        n = min(self.size, batch_size)
        idx = np.random.choice(self.size, size=n, replace=False)
        B = {k: v[idx] for k, v in self.buffer.items()}
        return B
        
    def get_statistics(self):
        if self.size == 0:
            return {"buffer/size": 0}
        
        r = self.buffer["reward"][:self.size]
        w = self.buffer["weight_assist"][:self.size]
        a = self.buffer["action"][:self.size]

        action_norm = np.linalg.norm(a, axis=1)
        weighted_action_norm = (w * action_norm).mean()

        return {
            "buffer/size": int(self.size),
            "buffer/fill_ratio": float(self.size / self.capacity),
            "buffer/reward_mean": float(r.mean()),
            "buffer/reward_std": float(r.std() + 1e-8),
            "buffer/weight_assist_mean": float(w.mean()),
            "buffer/weight_assist_nonzero_ratio": float((w > 1e-3).mean()),
            "buffer/action_norm_mean": float(action_norm.mean()),
            "buffer/weighted_action_norm_mean": float(weighted_action_norm),
        }