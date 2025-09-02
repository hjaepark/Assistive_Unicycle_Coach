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
from ray.tune import CheckpointConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.stopper import Stopper, CombinedStopper
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
        checkpoint_score_attribute="evaluation/custom_metrics/success_mean",
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
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .framework("torch")  
        .environment(
            "UnicyclePendulumPD-v0",
            env_config={}
        )
        .callbacks(SuccessCallback)
        .debugging(seed=SEED)
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            rollout_fragment_length=1500,
        )
        .training(
            train_batch_size=2048,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 1_000_000,
            },
            gamma=0.99,
            actor_lr=3e-4,
            critic_lr=3e-4,
            policy_model_config={"fcnet_hiddens": [256, 256]},
            q_model_config={"fcnet_hiddens": [256, 256]},
            num_steps_sampled_before_learning_starts=2048,
        )
        .resources(num_gpus=NUM_GPUS)
        .evaluation(
            evaluation_interval=EVAL_INTERVAL,
            evaluation_duration=EVAL_DURATION,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=False,
            evaluation_num_env_runners=1,
            evaluation_config={"explore": False},
        )
    )
    cfg.min_sample_timesteps_per_iteration = 1500

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
        metric="evaluation/custom_metrics/success_mean",
        mode="max",
    )

    mean_return = (best_result.metrics or {}).get("evaluation", {}).get("episode_reward_mean", float("nan"))
    print(f"\n[info] Best reward: {mean_return:.2f}")

    if best_result.checkpoint:
        best_model_path = run_dir / "best_model"
        if best_model_path.exists():
            shutil.rmtree(best_model_path)
        best_result.checkpoint.to_directory(best_model_path.as_posix())
        print(f"[info] Best model copied to: {best_model_path}")
    ray.shutdown()

if __name__ == "__main__":
    main()

