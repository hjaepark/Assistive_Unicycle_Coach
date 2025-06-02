# train/unicycle_train/student/train.py
# RLlib 2.12.0 – legacy ("old") stack

"""
Train a standalone student SAC agent on the Mujoco *UnicyclePendulumPD‑v0* task.
"""

import datetime
import shutil
from pathlib import Path
from typing import Any, Dict

import ray
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.evaluation.episode import Episode

from train.unicycle_train import path as P

# ======================================================================
# (1) environment factory – student training
# ======================================================================

def unicycle_env_creator(env_config: Dict[str, Any] | None = None):
    """Gymnasium factory registered for RLlib."""
    import gymnasium as gym, envs.unicycle 

    env_config = env_config or {}
    return gym.make("UnicyclePendulumPD-v0", **env_config)


tune.register_env("UnicycleStudentRLlib", unicycle_env_creator)

# ----------------------------------------------------------------------
# training‑termination criterion
# ----------------------------------------------------------------------
TARGET_SUCCESS: int = 200  # minimum number of successes per iteration to stop

# ======================================================================
# (2) callback – extra metrics + early stop when TARGET_SUCCESS reached
# ======================================================================

class MetricAndStopCallback(DefaultCallbacks):
    """Logs custom metrics and triggers Tune's termination condition."""

    # -------------- RLlib hooks --------------
    def on_episode_end(self, *, episode: Episode, **__):
        info = episode.last_info_for()
        if info:
            episode.custom_metrics.update(
                num_success=info.get("num_success", 0),
                num_falls=info.get("num_falls", 0),
                x_final=info.get("x_position", 0),
            )

    def on_train_result(self, *, result: dict, **__):
        it = result["training_iteration"]
        rew = result["episode_reward_mean"]
        succ = result["custom_metrics"].get("num_success_max", 0)
        falls = result["custom_metrics"].get("num_falls_max", 0)
        dist = result["custom_metrics"].get("x_final_mean", 0)

        print(
            f"[it={it:>4}]  reward={rew:8.2f}  success={succ}  "
            f"falls={falls}  x̄={dist:5.2f} "
        )

        # ---------- early stopping ----------
        if succ >= TARGET_SUCCESS:
            print(f"[Stop] {TARGET_SUCCESS} successes reached → terminating.")
            result["done"] = True  # consumed by Tune's stopper

# ======================================================================
# (3) SAC configuration
# ======================================================================

sac_cfg = (
    SACConfig()
    .environment("UnicycleStudentRLlib")
    .framework("torch")
    .resources(num_gpus=1)
    .callbacks(MetricAndStopCallback)
).to_dict()

# ======================================================================
# (4) launcher – Tune driver + checkpoint export
# ======================================================================

def main() -> None:
    logs_root = P.LOGS_STUDENT_DIR
    run_name = f"unicycle_student_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    tuner = tune.Tuner(
        "SAC",
        param_space=sac_cfg,
        run_config=RunConfig(
            name=run_name,
            storage_path=str(logs_root),
            stop={"timesteps_total": 10**7},
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=1,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max",
                num_to_keep=2,
            ),
        ),
    )

    results = tuner.fit()
    best = results.get_best_result()
    if best:
        src = best.checkpoint.path
        dst_root = P.EXPERT_MODEL_DIR
        dst_root.mkdir(parents=True, exist_ok=True)

        dst = dst_root / Path(src).name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"✓ Best checkpoint copied to {dst}")


if __name__ == "__main__":
    ray.shutdown()
    main()