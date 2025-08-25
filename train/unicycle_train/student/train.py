from typing import Any, Dict
from pathlib import Path

import ray
from ray import tune
from ray.air import CheckpointConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    EPISODE_RETURN_MAX,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_EPISODES_LIFETIME,
)

import gymnasium as gym
import envs.unicycle
from train.unicycle_train import path as P

def unicycle_env_creator(env_config: Dict[str, Any] | None = None):
    import envs.unicycle

    env_config = env_config or {}
    env = gym.make("UnicyclePendulumPD-v0", **env_config)
    return env

def next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    nums = []
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("run"):
            s = d.name[3:]
            if s.isdigit():
                nums.append(int(s))
    n = (max(nums) + 1) if nums else 1
    rd = base / f"run{n}"
    rd.mkdir(parents=True, exist_ok=False)
    return rd

def main():
    ray.init()
    
    run_dir = next_run_dir(P.LOGS_STUDENT_DIR)
    print(f"[info] run_dir: {run_dir}")

    tune.register_env("UnicyclePendulumPD-v0", unicycle_env_creator)

    config = {
        "env": "UnicyclePendulumPD-v0",
        "env_config": {},
        "framework": "torch",
        "evaluation_interval": 5,
        "evaluation_duration": 5,
        "evaluation_duration_unit": "episodes",
        "evaluation_num_env_runners": 1,
        "evaluation_parallel_to_training": True,
    }

    checkpoint_config = CheckpointConfig(
        checkpoint_frequency=10,
        checkpoint_at_end=True,
        num_to_keep=1,
        checkpoint_score_attribute="env_runners/episode_return_mean",
        checkpoint_score_order="max",
    )

    analysis = tune.run(
        "SAC",
        config=config,
        stop={"training_iteration": 50},
        storage_path=str(run_dir),
        checkpoint_config=checkpoint_config,
        verbose=1,
    )

    best_trial = analysis.get_best_trial(
        metric="env_runners/episode_return_mean",
        mode="max"
    )

    if best_trial:
        print(f"\n [info] Best reward: {best_trial.last_result['env_runners']['episode_return_mean']:.2f}")
        print(f"[info] Checkpoint path: {best_trial.checkpoint.path}")
        if best_trial.checkpoint:
            import shutil
            best_model_path = run_dir / "best_model"
            if best_model_path.exists():
                shutil.rmtree(best_model_path)
            shutil.copytree(best_trial.checkpoint.path, best_model_path)
            print(f"[info] Best model copied to: {best_model_path}")
            
    ray.shutdown()

if __name__ == "__main__":
    main()
