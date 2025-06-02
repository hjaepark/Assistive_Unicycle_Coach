# train/unicycle_train/coach/reward_based_skill_estimation.py
# RLlib 2.12.0 – legacy (“old”) stack

"""
Train a “coach” SAC agent that linearly blends the student’s action with a fixed expert policy.  
The blend weight w is updated online from a sliding reward window so that the expert takes over when the student under-performs.
"""

import datetime
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import ray
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.evaluation.episode import Episode

from train.unicycle_train import path as P

R_MAX_CONST: float = 350.0          # nominal expert return
WINDOW: int = 10                    # episodes averaged for skill estimate
INIT_W: float = 0.5                 # start with 50 % expert assistance

# ────────────────────────────────────────────────────────────────
# (1) environment factory – shared by student & coach
# ────────────────────────────────────────────────────────────────
def unicycle_env_creator(env_config: Dict[str, Any]):
    """Gymnasium factory registered for RLlib."""
    import gymnasium as gym, envs.unicycle
    return gym.make("UnicyclePendulumPD-v0", **env_config)


tune.register_env("UnicycleStudentRLlib", unicycle_env_creator)

tune.register_env(
    "UnicycleCoachStudentRLlib",
    lambda cfg: __import__(
        "train.unicycle_train.coach.action_blending_wrapper",
        fromlist=["ActionBlendingWrapper"],
    ).ActionBlendingWrapper(cfg),
)

# ────────────────────────────────────────────────────────────────
# (2)  callback factory – online skill estimation
# ────────────────────────────────────────────────────────────────
def SkillEstimationCallbackFactory(r_max: float, window: int = WINDOW):
    """Returns a subclass of DefaultCallbacks that maintains w."""

    class _SkillCB(DefaultCallbacks):
        """Internal RLlib callback used by the coach experiment."""

        def __init__(self):
            super().__init__()
            self.rew_buf: deque[float] = deque(maxlen=window)
            self.r_max = r_max

        # --------------------------------------------------
        # RLlib hooks
        # --------------------------------------------------
        def on_episode_end(
            self,
            *,
            worker,
            base_env,
            episode: Episode,
            env_index: int,
            **__,
        ):
            # 1) log common metrics pulled from env.info
            info = episode.last_info_for()
            if info:
                episode.custom_metrics.update(
                    num_success=info.get("num_success", 0),
                    num_falls=info.get("num_falls", 0),
                    x_final=info.get("x_position", 0),
                )

            # 2) update sliding-window reward buffer
            self.rew_buf.append(episode.total_reward)
            if len(self.rew_buf) < self.rew_buf.maxlen:
                return  # warm-up period

            r_avg = float(np.mean(self.rew_buf))
            w_new = float(np.clip(1.0 - r_avg / self.r_max, 0.0, 1.0))

            # broadcast new weight to **all** underlying envs
            for env in base_env.get_sub_environments():
                if hasattr(env, "set_weight"):
                    env.set_weight(w_new)

            episode.custom_metrics.update(w=w_new, r_avg=r_avg)

        def on_train_result(self, *, result: dict, **__):
            """Pretty CLI logging every training iteration."""
            it = result["training_iteration"]
            rew = result["episode_reward_mean"]
            wcur = result["custom_metrics"].get("w_mean", np.nan)
            succ = result["custom_metrics"].get("num_success_max", 0)
            falls = result["custom_metrics"].get("num_falls_max", 0)
            dist = result["custom_metrics"].get("x_final_mean", 0)
            print(
                f"[it={it:>4}]  r_max={self.r_max:5.1f}  reward={rew:8.2f}  "
                f"current-w={wcur:5.3f}  success={succ}  falls={falls}  "
                f"x̄={dist:5.2f}"
            )

    return _SkillCB


# ────────────────────────────────────────────────────────────────
# (3)  main launcher
# ────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"✓ r_max set to {R_MAX_CONST}")

    env_conf = {
        "env_cfg": {},
        "expert_ckpt": P.EXPERT_MODEL_DIR,
        "initial_w": INIT_W,
    }

    sac_cfg = (
        SACConfig()
        .environment("UnicycleCoachStudentRLlib", env_config=env_conf)
        .framework("torch")
        .resources(num_gpus=1)
        .callbacks(SkillEstimationCallbackFactory(R_MAX_CONST))
    ).to_dict()

    logs_root = P.LOGS_COACH_DIR
    run_name = (
        f"student_reward_based_skill_estimation_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    )

    tuner = tune.Tuner(
        "SAC",
        param_space=sac_cfg,
        run_config=RunConfig(
            name=run_name,
            storage_path=str(logs_root),
            stop={"timesteps_total": 500_000},
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=20,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max",
                num_to_keep=2,
            ),
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    ray.shutdown()
    main()
