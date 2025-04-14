# train/unicycle_train/evaluation.py

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
from gymnasium import spaces

import envs.unicycle as unicycle
from train.unicycle_train.monitor_callback.student_monitor import StudentMonitor
from train.unicycle_train.utils.path_utils import get_logs_dir

def run_evaluation(
    model_path: str,
    timesteps: int = 10_000,
    seed: int = 42,
    is_pre_test: bool = True
):
    """
    Run evaluation (pre-test or post-test) for a given model.
    
    Args:
        model_path: Path to the model to evaluate
        timesteps: Number of timesteps to run evaluation
        seed: Random seed for reproducibility
        is_pre_test: Whether this is a pre-test (before training) or post-test
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create and setup environment
    base_env = gym.make("Unicycle-v0")
    base_env = Monitor(base_env)
    
    # Get observation and action dimensions
    raw_obs_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]
    
    # Load model and modify its observation space
    model = SAC.load(model_path)
    low = np.concatenate([base_env.observation_space.low,
                         np.full((action_dim,), -1.0)])
    high = np.concatenate([base_env.observation_space.high,
                          np.full((action_dim,), 1.0)])
    model.policy.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Setup monitoring
    student_monitor = StudentMonitor()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_type = "pre_test" if is_pre_test else "post_test"
    run_name = f"{test_type}_{timestamp}"

    # Create log directory
    base_log_dir = os.path.join(get_logs_dir(), f"{test_type}_logs")
    os.makedirs(base_log_dir, exist_ok=True)
    log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Configure logger with tensorboard and stdout
    logger = configure(log_dir, ["tensorboard", "stdout"])
    student_monitor._logger = logger

    print(f"[{test_type.title()}] model={model_path}")
    print(f"Timesteps={timesteps}, logs={log_dir}")

    # Run evaluation
    global_step = 0
    episode_reward = 0.0
    episode_count = 0
    total_reward_sum = 0.0

    obs, _ = base_env.reset()
    # Extend observation with zeros
    dummy_obs = np.zeros(action_dim, dtype=np.float32)
    obs = np.concatenate([obs, dummy_obs], axis=0)
    
    done = False
    truncated = False

    while global_step < timesteps:
        action, _ = model.predict(obs, deterministic=True)
        next_obs_raw, reward, done, truncated, info = base_env.step(action)
        info["reward"] = reward
        info["student_action"] = action

        # Extend next observation with zeros
        dummy_obs = np.zeros(action_dim, dtype=np.float32)
        next_obs = np.concatenate([next_obs_raw, dummy_obs], axis=0)

        episode_reward += reward
        callback_locals = dict(infos=[info])
        student_monitor.locals = callback_locals
        student_monitor._on_step()
        logger.dump(step=global_step)
        global_step += 1

        if done or truncated:
            total_reward_sum += episode_reward
            episode_count += 1
            obs, _ = base_env.reset()
            # Extend observation with zeros
            dummy_obs = np.zeros(action_dim, dtype=np.float32)
            obs = np.concatenate([obs, dummy_obs], axis=0)
            episode_reward = 0.0
            done = False
            truncated = False
        else:
            obs = next_obs

    avg_reward = (total_reward_sum / episode_count) if episode_count else 0.0
    print(f"[{test_type.title()}] Done. Steps={global_step}, Eps={episode_count}, Avg={avg_reward:.4f}")
    base_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation (pre-test or post-test).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument("--timesteps", type=int, default=10000, help="Number of timesteps to run evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pre_test", action="store_true", help="Run pre-test (before training)")
    parser.add_argument("--post_test", action="store_true", help="Run post-test (after training)")
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    if args.pre_test and args.post_test:
        raise ValueError("Cannot run both pre-test and post-test at the same time")
    if not args.pre_test and not args.post_test:
        raise ValueError("Must specify either --pre_test or --post_test")

    run_evaluation(
        model_path=args.model_path,
        timesteps=args.timesteps,
        seed=args.seed,
        is_pre_test=args.pre_test
    ) 