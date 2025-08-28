# train/unicycle_train/student/train.py

from typing import Any, Dict
from pathlib import Path
import shutil

import os
import random
import torch
import numpy as np

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
from ray.tune.stopper import CombinedStopper
from train.unicycle_train.evaluate import SuccessCallback, EvalSuccessStopper

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
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    TARGET_SUCCESS_RATE = 0.95
    EVAL_INTERVAL = 20
    EVAL_DURATION = 100

    ray.init()

    run_dir = next_run_dir(P.LOGS_STUDENT_DIR)
    print(f"[info] run_dir: {run_dir}")

    tune.register_env("UnicyclePendulumPD-v0", unicycle_env_creator)

    checkpoint_config = CheckpointConfig(
        checkpoint_frequency=50,
        checkpoint_at_end=True,
        num_to_keep=1,
        checkpoint_score_attribute="evaluation/env_runners/success_rate",
        checkpoint_score_order="max",
    )

    stopper = CombinedStopper(
        EvalSuccessStopper(
            target_success_rate=TARGET_SUCCESS_RATE,
            minimum_number_of_episodes=EVAL_DURATION,
        )
    )

    try:
        NUM_GPUS = 1 if torch.cuda.is_available() else 0
    except Exception:
        NUM_GPUS = 0

    cfg = (
        SACConfig()
        .framework("torch")  
        .environment("UnicyclePendulumPD-v0", env_config={})
        .callbacks(SuccessCallback)
        .debugging(seed=SEED)
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=2,
            rollout_fragment_length=200,
        )
        .training(
            train_batch_size=8192,
            replay_buffer_config={
                "capacity": 1_000_000,
                "batch_size": 256,
                "prioritized_replay": False,
            },
            tau=0.005,
            gamma=0.99,
            target_entropy="auto",
            policy_model_config={"fcnet_hiddens": [256, 256]},
            q_model_config={"fcnet_hiddens": [256, 256]},
        )
        .learners(num_learners=1, num_gpus_per_learner=NUM_GPUS)
        .evaluation(
            evaluation_interval=EVAL_INTERVAL,
            evaluation_duration=EVAL_DURATION,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=True,
            evaluation_num_env_runners=1,
            evaluation_config={"explore": False,
                               "seed": None,
                               "env_config": {},},
        )
    )

    cfg.num_steps_sampled_before_learning_starts = 5000

    tuner = tune.Tuner(
        "SAC",
        run_config=tune.RunConfig(
            storage_path=str(run_dir),
            checkpoint_config=checkpoint_config,
            stop=stopper,
            verbose=1,
        ),
        param_space=cfg.to_dict(),
    )
    result_grid = tuner.fit()

    best_result = result_grid.get_best_result(
        metric="evaluation/env_runners/success_rate",
        mode="max",
    )

    eval_envr = (best_result.metrics or {}).get("evaluation", {}).get("env_runners", {})
    mean_ret = eval_envr.get("episode_return_mean", float("nan"))
    print(f"\n[info] Best reward: {mean_ret:.2f}")

    if best_result.checkpoint:
        best_model_path = run_dir / "best_model"
        if best_model_path.exists():
            shutil.rmtree(best_model_path)
        best_result.checkpoint.to_directory(best_model_path.as_posix())
        print(f"[info] Best model copied to: {best_model_path}")
    ray.shutdown()

if __name__ == "__main__":
    main()
