from typing import Any, Dict
from pathlib import Path
import os
import random
import torch
import numpy as np

import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from train.unicycle_train.evaluate import SuccessCallback

import gymnasium as gym
import envs.unicycle
from train.unicycle_train import path as P

def unicycle_env_creator(env_config: Dict[str, Any] | None = None):
    """RLlib이 환경을 생성할 때 사용하는 함수입니다."""
    import envs.unicycle
    env_config = env_config or {}
    env = gym.make("UnicyclePendulumPD-v0", **env_config)
    return env

def main():
    # --- 기본 설정: 시드 고정 및 Ray 초기화 ---
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ray.init()
    tune.register_env("UnicyclePendulumPD-v0", unicycle_env_creator)
    NUM_GPUS = 1 if torch.cuda.is_available() else 0

    cfg = (
        SACConfig()
        .framework("torch")
        .environment(
            "UnicyclePendulumPD-v0",
            # [FIX #1] 학습 정체 탈출: 에이전트가 '성공'을 쉽게 경험하도록 목표를 매우 가깝게 설정
            # 이를 통해 Fall_Rate=100%의 지역 최적해에서 탈출을 시도합니다.
            env_config={}
        )
        .callbacks(SuccessCallback)
        .debugging(seed=SEED)
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            rollout_fragment_length=256,
            # [FIX #2] 크래시 방지: RLlib 내부 데이터 처리 버그(크래시)를 우회하기 위한 설정
        )
        .training(
            train_batch_size=256,
            replay_buffer_config={
                # [FIX] Prioritized Replay Buffer를 계속 사용하되,
                # 우선순위가 0이 되는 것을 막기 위해 작은 epsilon 값을 추가합니다.
                "type": "EpisodeReplayBuffer",
                "capacity": 100_000,
            },
            gamma=0.99,
            actor_lr=3e-4,
            critic_lr=3e-4,
            policy_model_config={"fcnet_hiddens": [256, 256]},
            q_model_config={"fcnet_hiddens": [256, 256]},
        )
        .learners(num_gpus_per_learner=NUM_GPUS)
    )
    cfg.num_steps_sampled_before_learning_starts = 1000

    algo = cfg.build()

    print("\n--- Starting Minimal Baseline Training (Fixed) ---")
    print("Goal: Check if Fall_Rate drops below 100% and Episode_Reward shows a significant jump.")

    for i in range(1, 5001):
        try:
            result = algo.train()
            if i % 100 == 0:
                print(f"--- Iteration {i} finished ---")
        except Exception as e:
            print(f"An error occurred during training at iteration {i}: {e}")
            break # 오류 발생 시 루프 중단

    save_dir = algo.save(checkpoint_dir=str(P.LOGS_STUDENT_DIR / "test_run_model_fixed"))
    print(f"\n--- Training Finished ---")
    print(f"Final model checkpoint saved in: {save_dir}")

    ray.shutdown()

if __name__ == "__main__":
    main()


