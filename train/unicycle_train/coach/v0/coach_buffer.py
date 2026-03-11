# train/unicycle_train/coach/v0/coach_buffer.py

import numpy as np

class CoachReplayBuffer:
    def __init__(self, capacity=1_000_000, obs_dim=22, action_dim=2):
        """
        Args:
            capacity: max number of transitions (default 1M)
            obs_dim: coach observation dimension (state 19 + student_action 2 + r_env 1 = 22)
            action_dim: coach action dimension (2D assist force)
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.buffer = {
            "obs": np.zeros((capacity, obs_dim), dtype=np.float32),
            "action": np.zeros((capacity, action_dim), dtype=np.float32),
            "reward": np.zeros(capacity, dtype=np.float32),
            "next_obs": np.zeros((capacity, obs_dim), dtype=np.float32),
            "done": np.zeros(capacity, dtype=bool),
        }

        self.ptr = 0        # Current write position (circular)
        self.size = 0       # Number of valid transitions

        # Track the index of the last stored transition 
        # (for updating reward at inner loop end)
        self._last_stored_idx = None

    def add(self, obs, action, reward, next_obs, done):
        assert obs.shape == (self.obs_dim,), f"obs shape {obs.shape} != ({self.obs_dim},)"
        assert next_obs.shape == (self.obs_dim,), f"next_obs shape {next_obs.shape} != ({self.obs_dim},)"
        assert action.shape == (self.action_dim,), f"action shape {action.shape} != ({self.action_dim},)"

        i = self.ptr
        self.buffer["obs"][i] = obs
        self.buffer["action"][i] = action
        self.buffer["reward"][i] = reward
        self.buffer["next_obs"][i] = next_obs
        self.buffer["done"][i] = done

        self._last_stored_idx = i
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update_last_reward(self, reward):
        if self._last_stored_idx is not None:
            self.buffer["reward"][self._last_stored_idx] = reward

    def sample(self, batch_size=256):
        if self.size == 0:
            return {"size": 0}

        n = min(self.size, batch_size)
        idx = np.random.choice(self.size, size=n, replace=False)
        return {k: v[idx] for k, v in self.buffer.items()}

    def get_statistics(self):
        if self.size == 0:
            return {"buffer/size": 0}

        r = self.buffer["reward"][:self.size]
        a = self.buffer["action"][:self.size]
        action_norm = np.linalg.norm(a, axis=1)

        # Count non-zero rewards (only last steps of inner loops have reward)
        nonzero_rewards = r[r != 0.0]

        return {
            "buffer/size": int(self.size),
            "buffer/fill_ratio": float(self.size / self.capacity),
            "buffer/reward_mean": float(r.mean()),
            "buffer/reward_nonzero_count": int(len(nonzero_rewards)),
            "buffer/reward_nonzero_mean": float(nonzero_rewards.mean()) if len(nonzero_rewards) > 0 else 0.0,
            "buffer/action_norm_mean": float(action_norm.mean()),
        }