# student_train.py

import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import random
import torch

import envs.unicycle as unicycle
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

##########################################################################################################################
##########################################################################################################################
######################## Student agent ideal environment training code for the Unicycle Project ##########################
##########################################################################################################################
##########################################################################################################################

# 1) Basic directory setup
current_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
data_dir = os.path.join(current_dir, "data")

# Timestamp for run identification
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"student_{timestamp}"

# Detailed structure for models and logs
models_dir = os.path.join(data_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model", run_name)
final_model_dir = os.path.join(models_dir, "final_model")
logs_dir = os.path.join(data_dir, "logs", run_name)

# Create directories if not exist
directories = [data_dir, models_dir, best_model_dir, final_model_dir, logs_dir]
for d in directories:
    os.makedirs(d, exist_ok=True)

# 2) Callback for stopping training based on success count
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

# 3) Train function
def train(env, total_timesteps=10**10):

    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        device='cuda',
        tensorboard_log=logs_dir
    )

    eval_env = gym.make('Student-v0')
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

    final_model_path = os.path.join(final_model_dir, f"student_{timestamp}_final")
    model.save(final_model_path)
    print(f"Training finished. Final model saved at: {final_model_path}")

# 4) Test function
def test(env, path_to_model):

    model = SAC.load(path_to_model)
    print(f"Loaded model from: {path_to_model}")

    obs = env.reset()
    done = False
    truncated = False

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            if done or truncated:
                obs = env.reset()
                done = False
                truncated = False

            if env.render_mode == 'human':
                env.render()

    except KeyboardInterrupt:
        print("Test interrupted (Ctrl + C).")
    finally:
        env.close()

# 5) Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the student model.')
    parser.add_argument('-train', '--train', action='store_true', help='Train the model.')
    parser.add_argument('-test', '--test', type=str, help='Test the model (path to model file).')
    parser.add_argument('--timesteps', type=int, default=10**10, help='Number of timesteps for training.')
    args = parser.parse_args()

    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Train mode
    if args.train:
        env = gym.make('Student-v0')
        train(env, total_timesteps=args.timesteps)
        env.close()

    # Test mode
    if args.test:
        test_env = gym.make('Student-v0', render_mode='human')
        test(test_env, args.test)
        test_env.close()
