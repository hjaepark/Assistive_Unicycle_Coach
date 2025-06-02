# train/unicycle_train/coach/action_blending_wrapper.py
# RLlib 2.12.0 – legacy ("old") stack

"""
Linearly blends the student's action a_S with an expert SAC policy action a_E
before forwarding the command to the underlying Mujoco Unicycle environment.

    a_blend = a_S + w · (a_E – a_S)

The blend weight w ∈ [0,1] is externally adjustable through set_weight.
The environment reward r_env is returned unmodified.
"""

import numpy as np
import gymnasium as gym
from typing import Any, Dict

from ray.rllib.algorithms.sac import SAC

from train.unicycle_train import path as P

# ────────────────────────────────────────────────────────────────
# (1) environment factory – shared by student & coach
# ────────────────────────────────────────────────────────────────

def unicycle_env_creator(env_cfg: Dict[str, Any] | None = None):
    """Gymnasium factory registered for RLlib."""
    import envs.unicycle  # registers "UnicyclePendulumPD-v0"

    env_cfg = env_cfg or {}
    return gym.make("UnicyclePendulumPD-v0", **env_cfg)


# ────────────────────────────────────────────────────────────────
# (2) action‑blending wrapper
# ────────────────────────────────────────────────────────────────

class ActionBlendingWrapper(gym.Wrapper):
    """Wrapper that performs linear action blending with an expert SAC agent."""

    # ------------------------------------------------------------
    # constructor & helpers
    # ------------------------------------------------------------

    def __init__(self, env_config: Dict[str, Any] | None = None):
        env_config = env_config or {}
        inner_cfg = env_config.get("env_cfg", {})
        super().__init__(unicycle_env_creator(inner_cfg))

        # expert SAC policy ---------------------------------------------------
        ckpt_path = env_config.get("expert_ckpt", P.EXPERT_MODEL_DIR)
        self.expert = SAC.from_checkpoint(str(ckpt_path))

        # initial blend weight ----------------------------------------------
        self.weight: float = float(env_config.get("initial_w", 1.0))

    # ------------------------------------------------------------
    # external control – setter / getter
    # ------------------------------------------------------------

    def set_weight(self, w: float) -> None:
        """Update *w* in the closed interval [0,1]."""
        self.weight = float(np.clip(w, 0.0, 1.0))

    def get_weight(self) -> float:
        """Return current blend weight."""
        return self.weight

    # ------------------------------------------------------------
    # Gymnasium API overrides
    # ------------------------------------------------------------

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, student_action):
        """Blend *student_action* with expert action and step the env."""
        # 1) expert action ----------------------------------------------------
        raw = self.expert.compute_single_action(self.env.unwrapped._get_obs(), explore=False)
        a_E = raw[0] if isinstance(raw, tuple) else raw
        a_E = np.asarray(a_E, np.float32).reshape(-1)

        # 2) student action ---------------------------------------------------
        a_S = np.asarray(student_action, np.float32).reshape(-1)

        # 3) ensure equal dimensionality -------------------------------------
        target_dim = max(2, a_S.size)  # 4‑DoF student → 4, otherwise 2

        def _fit(v: np.ndarray) -> np.ndarray:
            if v.size < target_dim:
                return np.pad(v, (0, target_dim - v.size), constant_values=0.0)
            return v[:target_dim]

        a_E = _fit(a_E)
        a_S = _fit(a_S)

        # 4) blend first 2 dimensions only -----------------------------------
        a_blend = a_S.copy()
        a_blend[:2] = a_S[:2] + self.weight * (a_E[:2] - a_S[:2])

        # 5) clip to action space --------------------------------------------
        low, high = (
            (self.action_space.low, self.action_space.high)
            if self.action_space.shape[0] == a_blend.size
            else (-1.0, 1.0)
        )
        a_final = np.clip(a_blend, low, high)

        # 6) step environment -------------------------------------------------
        obs, r_env, term, trunc, info = self.env.step(a_final)

        info.update({
            "w": self.weight,
            "a_S": a_S,
            "a_E": a_E,
            "a_F": a_final,
        })
        return obs, r_env, term, trunc, info

    # pass‑through helpers ----------------------------------------------------

    def render(self, *args, **kwargs):  
        return self.env.render(*args, **kwargs)

    def close(self): 
        self.env.close()
