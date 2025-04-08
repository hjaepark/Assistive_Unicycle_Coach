# train/unicycle_train/pre_test.py

import os
import argparse
import datetime
import numpy as np
import random
import torch

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

import envs.unicycle as unicycle
from train.unicycle_train.wrapper.dummy_action_merging_wrapper import DummyActionMergeWrapper
from train.unicycle_train.monitor_callback.student_monitor import StudentMonitor
from train.unicycle_train.monitor_callback.coach_monitor import CoachMonitor
from train.unicycle_train.utils.path_utils import get_pretest_logs_dir

def run_pre_test(
    untrained_model_path: str,
    timesteps: int = 10_000,
    seed: int = 42
):
    """
    Pre-test: alpha=0 => purely student. Logs data to TB.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_env = gym.make("Unicycle-v0")
    base_env = Monitor(base_env)
    env = DummyActionMergeWrapper(env=base_env, alpha=0.0, include_coach_in_obs=True)

    model = SAC.load(untrained_model_path)
    model.set_env(env)

    student_monitor = StudentMonitor()
    coach_monitor = CoachMonitor()

    base_pretest_log_dir = get_pretest_logs_dir()
    os.makedirs(base_pretest_log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_pretest_log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    logger = configure(log_dir, ["tensorboard", "stdout"])
    student_monitor._logger = logger
    coach_monitor._logger = logger

    print(f"[Pre-Test] model={untrained_model_path}, alpha=0.0")
    print(f"Timesteps={timesteps}, logs={log_dir}")

    global_step = 0
    episode_reward = 0.0
    episode_count = 0
    total_reward_sum = 0.0

    obs, _ = env.reset()
    done = False
    truncated = False

    while global_step < timesteps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        info["reward"] = reward

        episode_reward += reward
        callback_locals = dict(infos=[info])
        student_monitor.locals = callback_locals
        coach_monitor.locals = callback_locals

        student_monitor._on_step()
        coach_monitor._on_step()
        logger.dump(step=global_step)
        global_step += 1

        if done or truncated:
            total_reward_sum += episode_reward
            episode_count += 1
            obs, _ = env.reset()
            episode_reward = 0.0
            done = False
            truncated = False

    avg_reward = (total_reward_sum / episode_count) if episode_count else 0.0
    print(f"[Pre-Test] Done. Steps={global_step}, Eps={episode_count}, Avg={avg_reward:.4f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-test with alpha=0 rollout.")
    parser.add_argument("--untrained_model_path", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.untrained_model_path):
        raise FileNotFoundError(f"Untrained model not found: {args.untrained_model_path}")

    run_pre_test(
        untrained_model_path=args.untrained_model_path,
        timesteps=args.timesteps,
        seed=args.seed
    )
