# train/unicycle_train/train/fixed_alpha_train.py

import os
import argparse
import datetime
import numpy as np
import random
import torch

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import envs.unicycle as unicycle
from train.unicycle_train.wrapper.action_merging_wrapper import ActionMergeWrapper
from train.unicycle_train.monitor_callback.student_monitor import StudentMonitor
from train.unicycle_train.monitor_callback.coach_monitor import CoachMonitor
from train.unicycle_train.utils.path_utils import get_logs_dir, get_models_dir

def run_fixed_alpha_training(
    untrained_student_path: str,
    coach_model_path: str,
    total_timesteps: int = 1_000_000,
    alpha: float = 0.5
):
    """
    Train with a fixed alpha for merging (student vs coach).
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"fixed_alpha_{timestamp}"

    # Create log dirs
    base_log_dir = os.path.join(get_logs_dir(), "fixed_alpha_train_logs")
    os.makedirs(base_log_dir, exist_ok=True)
    logs_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(logs_dir, exist_ok=True)

    best_model_dir = os.path.join(get_models_dir(), "best_model", run_name)
    os.makedirs(best_model_dir, exist_ok=True)
    final_model_dir = os.path.join(get_models_dir(), "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    # Train env
    base_env = gym.make("Unicycle-v0")
    base_env = Monitor(base_env)
    train_env = ActionMergeWrapper(
        env=base_env,
        coach_model_path=coach_model_path,
        alpha=alpha
    )

    # Eval env
    eval_base_env = gym.make("Unicycle-v0")
    eval_base_env = Monitor(eval_base_env)
    eval_env = ActionMergeWrapper(
        env=eval_base_env,
        coach_model_path=coach_model_path,
        alpha=alpha
    )

    # Load untrained
    student_model = SAC.load(untrained_student_path, env=train_env)
    student_model.set_env(train_env)
    student_model.tensorboard_log = logs_dir  # TB root folder

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=logs_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    student_monitor = StudentMonitor()
    coach_monitor = CoachMonitor()
    callback_list = [eval_callback, student_monitor, coach_monitor]

    # Train
    student_model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        tb_log_name=run_name
    )

    # Save final
    final_model_path = os.path.join(final_model_dir, f"fixed_alpha_{timestamp}_final")
    student_model.save(final_model_path)
    print(f"[Train] Finished. Final model saved at: {final_model_path}")

def run_fixed_alpha_test(
    trained_student_path: str,
    coach_model_path: str,
    alpha: float = 0.5,
    render: bool = False
):
    """
    Quick test for a trained model with fixed alpha.
    """
    env_kwargs = {}
    if render:
        env_kwargs["render_mode"] = "human"

    base_env = gym.make("Unicycle-v0", **env_kwargs)
    base_env = Monitor(base_env)
    test_env = ActionMergeWrapper(
        env=base_env,
        coach_model_path=coach_model_path,
        alpha=alpha
    )

    model = SAC.load(trained_student_path, env=test_env)
    model.set_env(test_env)

    print(f"[Test] Loaded model: {trained_student_path}, alpha={alpha}")
    obs, _ = test_env.reset()
    done = False
    truncated = False

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            if done or truncated:
                obs, _ = test_env.reset()
                done = False
                truncated = False
    except KeyboardInterrupt:
        print("[Test] Interrupted.")
    finally:
        test_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed Alpha Training/Testing Script")
    parser.add_argument("--train", action="store_true", help="Train mode")
    parser.add_argument("--test", action="store_true", help="Test mode")

    parser.add_argument("--untrained_student_path", type=str, default=None)
    parser.add_argument("--trained_student_path", type=str, default=None)
    parser.add_argument("--coach_model_path", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.coach_model_path is None:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..'))
        default_coach = os.path.join(project_root, "pre_trained_models", "unicycle_trained.zip")
        args.coach_model_path = default_coach

    if args.train:
        run_fixed_alpha_training(
            untrained_student_path=args.untrained_student_path,
            coach_model_path=args.coach_model_path,
            total_timesteps=args.timesteps,
            alpha=args.alpha
        )

    if args.test:
        run_fixed_alpha_test(
            trained_student_path=args.trained_student_path,
            coach_model_path=args.coach_model_path,
            alpha=args.alpha,
            render=args.render
        )
