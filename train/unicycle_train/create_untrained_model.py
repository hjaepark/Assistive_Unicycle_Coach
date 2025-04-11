# train/unicycle_train/wrapper/create_untrained_model.py

import os
import argparse
import datetime
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import numpy as np
import random
import torch

import envs.unicycle as unicycle
from train.unicycle_train.wrapper.dummy_action_merging_wrapper import DummyActionMergeWrapper
from train.unicycle_train.utils.path_utils import get_untrained_dir

def create_untrained_student_model(save_path, seed=42):
    """
    Creates an untrained SAC model that sees (obs + coach_action) dimension.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_env = gym.make("Unicycle-v0")
    base_env = Monitor(base_env)
    env = DummyActionMergeWrapper(
        env=base_env,
        alpha=0.0,
        include_coach_in_obs=True
    )
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        device="cuda"
    )
    model.save(save_path)
    print(f"[create_untrained_model] Untrained student model saved: {save_path}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an untrained SAC model.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    untrained_dir = get_untrained_dir()
    os.makedirs(untrained_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.zip"
    save_path = os.path.join(untrained_dir, filename)
    create_untrained_student_model(save_path=save_path, seed=args.seed)
