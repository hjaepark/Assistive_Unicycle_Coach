# train/unicycle_train/coach/train.py
# RLlib 2.12.0 – legacy (“old”) stack
import datetime
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec

from train.unicycle_train import path as P
from train.unicycle_train.coach.action_blending_wrapper import ActionBlendingWrapper


# -----------------------------------------------------------------------------
# 1. Environment registration helpers
# -----------------------------------------------------------------------------

def unicycle_env_creator(env_config: Dict[str, Any]):
    """Factory used by the *student‑only* expert checkpoint."""
    import envs.unicycle  # noqa: F401 – side‑effect registers the Mujoco env
    return gym.make("UnicyclePendulumPD-v0", **env_config)

# Keep the legacy name so the expert checkpoint can still be loaded.
tune.register_env("UnicycleStudentRLlib", unicycle_env_creator)

# -----------------------------------------------------------------------------
# 2. Multi‑agent wrapper (student + coach)
# ---------------------------------------------------------X--------------------

class CoachStudentMAEnv(MultiAgentEnv):
    """Two‑agent interface in RLlib format.

    * **student** – outputs the raw action for the underlying Mujoco env.
    * **coach**   – outputs a scalar *w* in [0, 1] that blends the student
      action with a fixed expert SAC policy inside ActionBlendingWrapper.
    """

    def __init__(self, env_config: Dict[str, Any] | None = None):
        super().__init__()
        env_config = env_config or {}

        # Inner single‑agent environment (handles blending + Mujoco sim)
        self.inner = ActionBlendingWrapper(env_config)

        # RLlib >= 2.3 expects an explicit agent‑id set.
        self._agent_ids = {"student", "coach"}

        # ------------------------------------------------------------------
        # Observation / action spaces
        # ------------------------------------------------------------------
        state_space = self.inner.observation_space          # (17,)
        w_space     = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            "student": state_space,
            # coach sees the same 17‑D state *plus* its last chosen w  → 18‑D
            "coach":   gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(state_space.shape[0] + 1,),
                dtype=np.float32,
            ),
        })

        self.action_space = gym.spaces.Dict({
            "student": self.inner.action_space,
            "coach":   w_space,   # scalar in [0, 1]
        })

        # Reward coefficients for the coach: r_c = k1 * r_env - k2 * w
        self.k1 = float(env_config.get("kappa1", 1.0))
        self.k2 = float(env_config.get("kappa2", 0.05))

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: None | int = None, options: None | dict = None):
        obs_raw, info_raw = self.inner.reset(seed=seed, options=options)

        w = self.inner.get_weight()
        student_obs = np.asarray(obs_raw, dtype=np.float32)
        coach_obs   = np.concatenate([student_obs, [w]]).astype(np.float32)

        obs_dict = {"student": student_obs, "coach": coach_obs}
        info_dict = {"student": info_raw, "coach": {}}
        return obs_dict, info_dict

    def step(self, action_dict: dict):
        # ------------------------------------------------------------------
        # 1) split actions
        # ------------------------------------------------------------------
        a_student = action_dict["student"]
        w_raw     = action_dict["coach"]
        w         = float(w_raw.item() if isinstance(w_raw, np.ndarray) else w_raw)
        self.inner.set_weight(w)

        # ------------------------------------------------------------------
        # 2) inner env step  →  (obs, r_env, term, trunc, info)
        # ------------------------------------------------------------------
        obs, r_env, term, trunc, info = self.inner.step(a_student)

        # ------------------------------------------------------------------
        # 3) multi‑agent packaging
        # ------------------------------------------------------------------
        r_student = r_env
        r_coach   = self.k1 * r_env - self.k2 * w

        terminated = bool(term)
        truncated  = bool(trunc)

        student_obs = np.asarray(obs, dtype=np.float32)
        coach_obs   = np.concatenate([student_obs, [w]]).astype(np.float32)

        obs_dict = {"student": student_obs, "coach": coach_obs}
        rew_dict = {"student": r_student,  "coach": r_coach}
        terminated_dict = {"student": terminated, "coach": terminated, "__all__": terminated}
        truncated_dict  = {"student": truncated,  "coach": truncated,  "__all__": truncated}

        info_student = dict(info)
        info_student["w"] = w
        info_dict = {"student": info_student, "coach": {}}

        return obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict


# Register with Tune so the config can reference it by name.
tune.register_env("UnicycleCoachStudentRLlib", lambda cfg: CoachStudentMAEnv(cfg))

# -----------------------------------------------------------------------------
# 3. Custom callback for pretty CLI logging
# -----------------------------------------------------------------------------

class TrainLogger(DefaultCallbacks):
    """Pushes per‑episode stats to TensorBoard & prints nice console output."""

    def on_episode_end(self, *, episode, **__):
        info = episode.last_info_for("student")
        if not info:
            return
        episode.custom_metrics.update(
            num_success=info.get("num_success", 0),
            num_falls=info.get("num_falls", 0),
            x_final=info.get("x_position", 0.0),
        )

    def on_train_result(self, *, result: dict, **__):
        it = result["training_iteration"]

        pr = result.get("policy_reward_mean", {})
        r_s = pr.get("student", float("nan"))
        r_c = pr.get("coach",   float("nan"))

        succ  = result["custom_metrics"].get("num_success_max", 0)
        falls = result["custom_metrics"].get("num_falls_max",   0)
        dist  = result["custom_metrics"].get("x_final_mean",    float("nan"))

        w_hist = result["hist_stats"].get("w", [])
        w_mean = float(np.mean(w_hist)) if w_hist else np.nan

        print(
            f"[it={it:>4}]  "
            f"reward_s={r_s:8.2f}  reward_c={r_c:8.2f}  "
            f"w̄={w_mean:5.3f}  success={succ}  falls={falls}  x̄={dist:5.2f}"
        )


# -----------------------------------------------------------------------------
# 4. RLlib SAC configuration builder
# -----------------------------------------------------------------------------

def build_config() -> Dict:
    dummy = CoachStudentMAEnv({
        "env_cfg": {}, "expert_ckpt": P.EXPERT_MODEL_DIR, "initial_w": 0.5,
    })
    obs_s, act_s = dummy.observation_space["student"], dummy.action_space["student"]
    obs_c, act_c = dummy.observation_space["coach"],   dummy.action_space["coach"]

    cfg = (
        SACConfig()
        .environment(
            env="UnicycleCoachStudentRLlib",
            env_config={
                "env_cfg": {},
                "expert_ckpt": P.EXPERT_MODEL_DIR,
                "initial_w": 0.8,
                "kappa1": 1.0,
                "kappa2": 0.05,
            },
        )
        .framework("torch")
        .resources(num_gpus=1)
        .training(train_batch_size=4096, gamma=0.99)
        .callbacks(TrainLogger)
    )

    cfg = cfg.multi_agent(
        policies={
            "student": PolicySpec(None, obs_s, act_s, {}),
            "coach":   PolicySpec(None, obs_c, act_c, {}),
        },
        policy_mapping_fn=lambda *args, **__: args[0],  # identity mapping
        policies_to_train=["student", "coach"],
    )
    return cfg.to_dict()


# -----------------------------------------------------------------------------
# 5. Main launcher
# -----------------------------------------------------------------------------

def main() -> None:
    print("✓ joint‑training student & coach (κ₁, κ₂) with full state for coach")

    sac_cfg = build_config()
    logs_dir = P.LOGS_COACH_DIR
    run_name = f"student_coach_joint_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    tuner = tune.Tuner(
        "SAC",
        param_space=sac_cfg,
        run_config=RunConfig(
            name=run_name,
            storage_path=str(logs_dir),
            stop={"timesteps_total": 1_000_000},
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=25,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max",
                num_to_keep=3,
            ),
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    ray.shutdown()
    main()
