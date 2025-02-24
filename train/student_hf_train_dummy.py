# student_hf_train.py

import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import random
import torch
import envs
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

##########################################################################################################################
##########################################################################################################################
#################### Student agent + human factor environment training code for the Unicycle Project #####################
##########################################################################################################################
##########################################################################################################################

# Basic directory structure setup
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(current_dir, "data")

# Create timestamp directory for current run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"student_hf_{timestamp}"

# Setup detailed directory structure for current run
models_dir = os.path.join(data_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model", run_name)
final_model_dir = os.path.join(models_dir, "final_model")
logs_dir = os.path.join(data_dir, "logs", run_name)

# Create all directories
directories = [
    data_dir, models_dir, best_model_dir, final_model_dir, logs_dir
]
for d in directories:
    os.makedirs(d, exist_ok=True)

class TrainingStopCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        
        for info in infos:
            if "num_success" in info:
                success_count = info["num_success"]

                self.logger.record("train/num_success", success_count)

                if success_count >= 100:
                    if self.verbose > 0:
                        print("[TrainingStopCallback] Reached 100 success counts.")
                    return False 
        return True


def train(env, total_timesteps, safe_coef, safe_margin, sensor_noise_std, action_delay):
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        device='cuda',
        tensorboard_log=logs_dir
    )

    eval_env = gym.make(
        'StudentHF-v0',
        safe_coef=safe_coef,
        safe_margin=safe_margin,
        sensor_noise_std=sensor_noise_std,
        action_delay=action_delay,
    )
    
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

    final_model_path = os.path.join(final_model_dir, f"student_hf_{timestamp}_final")
    model.save(final_model_path)
    print(f"Training finished. Final model saved at: {final_model_path}")


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test student model.')
    parser.add_argument('-train', '--train', action='store_true', help='Train the model.')
    parser.add_argument('-test', '--test', type=str, help='Test the model (path to model file).')
    parser.add_argument('--timesteps', type=int, default=10**10, help='Number of timesteps for training.')

    parser.add_argument('--safe_coef', type=float, default=0.02, help='Safe coefficient for HF reward shaping.')
    parser.add_argument('--safe_margin', type=float, default=0.3, help='Safe margin in radians.')
    parser.add_argument('--sensor_noise_std', type=float, default=0.0, help='Std dev for sensor noise.')
    parser.add_argument('--action_delay', type=int, default=3, help='Action delay steps.')

    args = parser.parse_args()

    # Seed
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if args.train:
        env = gym.make('StudentHF-v0',
                       safe_coef=args.safe_coef,
                       safe_margin=args.safe_margin,
                       sensor_noise_std=args.sensor_noise_std,
                       action_delay=args.action_delay
                       )
        

        train(env, args.timesteps, args.safe_coef, args.safe_margin,
            args.sensor_noise_std, args.action_delay)
        env.close()

    if args.test:
        test_env = gym.make('StudentHF-v0',
                            render_mode='human',
                            safe_coef=args.safe_coef,
                            safe_margin=args.safe_margin,
                            sensor_noise_std=args.sensor_noise_std,
                            action_delay=args.action_delay
                            )        
        test(test_env, args.test)
        test_env.close()