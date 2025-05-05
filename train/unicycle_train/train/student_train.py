# student_train.py

import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import random
import torch
import time

import envs.unicycle as unicycle
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from gymnasium import spaces
from gymnasium.core import Wrapper
from train.unicycle_train.wrapper.extended_observation_wrapper import ExtendedObservationWrapper
from train.unicycle_train.utils.path_utils import (
    get_models_dir,
    get_logs_dir,
    get_train_logs_dir
)

##########################################################################################################################
# 1) Basic directory setup
##########################################################################################################################
# Timestamp for run identification
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"unicycle_student_{timestamp}"  

# Detailed structure for models and logs
models_dir = get_models_dir()
best_model_dir = os.path.join(models_dir, "best_model", run_name)
final_model_dir = os.path.join(models_dir, "final_model")
logs_dir = os.path.join(get_train_logs_dir(), run_name)

# Create directories if not exist
directories = [models_dir, best_model_dir, final_model_dir, logs_dir]
for d in directories:
    os.makedirs(d, exist_ok=True)

# Set trained model directory to load and test
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
trained_models_dir = os.path.join(project_root, 'trained_models')

##########################################################################################################################
# 2) Callback for stopping training based on success count
##########################################################################################################################
class TrainingStopCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if "num_success" in info:
                success_count = info["num_success"]
                self.logger.record("train/num_success", success_count)

                if success_count >= 1000:
                    if self.verbose > 0:
                        print("[TrainingStopCallback] Reached 1000 success counts.")
                    return False
        return True

##########################################################################################################################
# 3) Train function
##########################################################################################################################
def train(env, total_timesteps=10**10):
    # Wrap the environment
    env = ExtendedObservationWrapper(env)

    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        device='cuda',
        tensorboard_log=logs_dir
    )

    # Create evaluation environment with extended observation space
    eval_env = ExtendedObservationWrapper(gym.make('Unicycle-v0'))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=logs_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    success_stop_callback = TrainingStopCallback(verbose=1)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, success_stop_callback]
    )

    final_model_path = os.path.join(final_model_dir, f"unicycle_student_{timestamp}_final")
    model.save(final_model_path)
    print(f"Training finished. Final model saved at: {final_model_path}")

##########################################################################################################################
# 4) Test function
##########################################################################################################################
def test(env, path_to_model):
    # Wrap the environment
    env = ExtendedObservationWrapper(env)

    model = SAC.load(path_to_model)
    print(f"Loaded model from: {path_to_model}")

    obs, _ = env.reset()
    done = False
    truncated = False

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            if done or truncated:
                obs, _ = env.reset()
                done = False
                truncated = False

            if env.render_mode == 'human':
                env.render()

    except KeyboardInterrupt:
        print("Test interrupted (Ctrl + C).")
    finally:
        env.close()

##########################################################################################################################
# 5) Main execution
##########################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the student model.')
    parser.add_argument('-train', '--train', action='store_true', help='Train the model.')
    parser.add_argument('-test', '--test', type=str, help='Test the model (path to model file).')
    parser.add_argument('--timesteps', type=int, default=10**10, help='Number of timesteps for training.')
    args = parser.parse_args()

    # Set seeds for reproducibility
    # seed = 42

    # Time based random seed
    seed = int(time.time())
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Train mode
    if args.train:
        env = gym.make('Unicycle-v0')
        train(env, total_timesteps=args.timesteps)
        env.close()

    # Test mode
    if args.test:
        test_env = gym.make('Unicycle-v0', render_mode='human')
        path_to_model = os.path.join(trained_models_dir, args.test)
        test(test_env, path_to_model)
        test_env.close()
