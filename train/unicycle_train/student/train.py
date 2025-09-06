# train/unicycle_train/student/train.py

from typing import Any, Dict, Callable
from pathlib import Path
import os
import random
import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed as sb3_set_seed

from train.unicycle_train.evaluate import EvalCallBack

import gymnasium as gym
import envs.unicycle
from train.unicycle_train import path as P

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

def make_train_env(log_dir: Path, seed: int):
    base_env = gym.make("UnicyclePendulumPD-v0")

    # Save data to CSV
    obs_dim = int(np.prod(base_env.observation_space.shape))
    obs_keys = tuple(f"obs_{i}" for i in range(obs_dim))
    extra_keys = (
        "timestamp",
        "num_steps",
        "num_episodes",
        "unicycle_roll", 
        "unicycle_pitch", 
        "unicycle_yaw", 
        "pendulum_ang",
        "num_success", 
        "num_falls",
        "reward", 
        "goal_reached",
        "is_fallen", 
    )
    env = Monitor(
        base_env,
        filename=str(log_dir / "monitor.csv"),
        info_keywords=obs_keys + extra_keys,
    )

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env

def make_eval_env() -> gym.Env:
    return gym.make("UnicyclePendulumPD-v0")

class LoggingCallback(BaseCallback):
    """Log metrics in real-time as episodes complete"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # Check if any episode ended
        if len(self.locals.get("dones", [])) > 0:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    
                    # Log everything immediately when episode ends
                    if "episode" in info:
                        self.logger.record("train/episode_reward", info["episode"]["r"])
                        self.logger.record("train/episode_length", info["episode"]["l"])
                    
                    # Log all custom metrics
                    if "unicycle_roll" in info:
                        self.logger.record("train/unicycle_roll", info["unicycle_roll"])
                    if "unicycle_pitch" in info:
                        self.logger.record("train/unicycle_pitch", info["unicycle_pitch"])
                    if "unicycle_yaw" in info:
                        self.logger.record("train/unicycle_yaw", info["unicycle_yaw"])
                    if "pendulum_ang" in info:
                        self.logger.record("train/pendulum_ang", info["pendulum_ang"])
                    if "x_position" in info:
                        self.logger.record("train/final_x_position", info["x_position"])
                    if "y_position" in info:
                        self.logger.record("train/final_y_position", info["y_position"])
                    if "goal_reached" in info:
                        self.logger.record("train/goal_reached", float(info["goal_reached"]))
                    if "is_fallen" in info:
                        self.logger.record("train/is_fallen", float(info["is_fallen"]))
                    
                    self.logger.dump(self.num_timesteps)
        
        return True

def main():
    def gen_seed() -> int:
        return int.from_bytes(os.urandom(4), "little")
    SEED = gen_seed()
    sb3_set_seed(SEED)


    # Evaluation settings
    TARGET_SUCCESS_RATE = 0.95
    EVAL_INTERVAL = 50_000
    EVAL_DURATION = 100
    REQUIRE_CONSECUTIVE = 3

    run_dir = next_run_dir(P.LOGS_STUDENT_DIR)
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir = run_dir / "models"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "seed.txt").write_text(f"{SEED}\n")
    print(f"[info] run_dir: {run_dir} | seed={SEED}")

    env = make_train_env(run_dir, SEED)

    model = SAC(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        seed=SEED,
        tensorboard_log=str(tensorboard_dir),
    )
    model.set_random_seed(SEED)

    logging_callback = LoggingCallback(verbose=0)

    evaluation_callback = EvalCallBack(
        make_env_fn=make_eval_env,
        eval_interval_steps=EVAL_INTERVAL,
        n_eval_episodes=EVAL_DURATION,
        target_success_rate=TARGET_SUCCESS_RATE,
        best_model_path=str(checkpoint_dir / "best_model"),
        deterministic=True,
        seed=None,                       
        verbose=1,
        require_consecutive=REQUIRE_CONSECUTIVE,
    )

    callbacks = [logging_callback, evaluation_callback]

    TOTAL_TIMESTEPS = int(1e9)
    print(f"[info] Training Initiated (maximum timesteps: {TOTAL_TIMESTEPS:,})")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

    model_path = checkpoint_dir / "student_agent"
    model.save(model_path.as_posix())
    print(f"[info] Model saved to: {model_path}")

    env.close()

if __name__ == "__main__":
    main()
