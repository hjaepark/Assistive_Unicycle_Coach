# coach_student_fixed_alpha_train.py

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

##########################################################################################################################
# 1) Basic directory setup
##########################################################################################################################
current_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
data_dir = os.path.join(current_dir, "data")

# Timestamp for run identification
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"unicycle_coach_student_fixed_alpha_{timestamp}"

# Detailed structure for models and logs
models_dir = os.path.join(data_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model", run_name)
final_model_dir = os.path.join(models_dir, "final_model")
logs_dir = os.path.join(data_dir, "logs", run_name)

# Create directories if not exist
directories = [data_dir, models_dir, best_model_dir, final_model_dir, logs_dir]
for d in directories:
    os.makedirs(d, exist_ok=True)

# Directory containing the pretrained "coach" model
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
trained_models_dir = os.path.join(project_root, 'trained_models')

##########################################################################################################################
# 2) Coach + Student Action Merging Wrapper
##########################################################################################################################
"""
Loads a pre-trained SAC policy
and merges its action with the student's action by a fixed alpha.
"""
class CoachStudentFixedAlphaWrapper(gym.Wrapper):
    """
    Loads a pre-trained coach SAC policy (file) and merges the student's action
    with the coach's action by a fixed alpha. The alpha value remains constant.
    """

    def __init__(self, env, coach_model_path, alpha=0.5, include_coach_in_obs=False):
        super().__init__(env)
        # Load the coach model
        self.coach_model = SAC.load(coach_model_path, env=None)
        self.alpha = alpha
        self.include_coach_in_obs = include_coach_in_obs

        # Check observation (obs) and action dimensions from the environment
        self.raw_obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Adjust the observation space depending on whether to include the coach's action
        if self.include_coach_in_obs:
            new_obs_dim = self.raw_obs_dim + self.action_dim
            low = np.concatenate([
                env.observation_space.low,
                np.full((self.action_dim,), -1.0)
            ])
            high = np.concatenate([
                env.observation_space.high,
                np.full((self.action_dim,), 1.0)
            ])
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.observation_space = env.observation_space

        self.action_space = env.action_space
        self._last_obs_raw = None

    def reset(self, **kwargs):
        # Reset the environment
        obs_raw, info = self.env.reset(**kwargs)
        self._last_obs_raw = obs_raw

        # Obtain the coach action and optionally concatenate it to the observation
        coach_action, _ = self.coach_model.predict(obs_raw, deterministic=True)

        if self.include_coach_in_obs:
            obs = np.concatenate([obs_raw, coach_action], axis=0)
        else:
            obs = obs_raw

        return obs, info

    def step(self, student_action):
        if self._last_obs_raw is None:
            raise RuntimeError("Wrapper must be reset() before calling step().")

        # Compute the coach action
        coach_action, _ = self.coach_model.predict(self._last_obs_raw, deterministic=True)
        
        # Coach action after alpha scheduling
        coach_action_prime = self.alpha * (coach_action - student_action)

        # Combine student and coach actions with a fixed alpha
        final_action = student_action + self.alpha * (coach_action - student_action)
        final_action = np.clip(final_action, -1.0, 1.0)

        # Step the environment
        next_obs_raw, reward, done, truncated, info = self.env.step(final_action)
        info["student_action"] = student_action  # record student action
        info["coach_action_prime"] = coach_action_prime
        info["coach_action"] = coach_action  # record coach action for monitoring

        # Obtain the next coach action and optionally concatenate it to the next observation
        next_coach_action, _ = self.coach_model.predict(next_obs_raw, deterministic=True)
        if self.include_coach_in_obs:
            next_obs = np.concatenate([next_obs_raw, next_coach_action], axis=0)
        else:
            next_obs = next_obs_raw

        self._last_obs_raw = next_obs_raw
        return next_obs, reward, done, truncated, info

##########################################################################################################################
# 3) Student Monitor Callback
##########################################################################################################################
class StudentMonitor(BaseCallback):
    """
    Logs information related to the student (roll, pitch, reward, etc.).
    """
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            # Log various environmental info if necessary
            if "unicycle_roll" in info:
                self.logger.record("student/unicycle_roll", info["unicycle_roll"])
            if "unicycle_pitch" in info:
                self.logger.record("student/unicycle_pitch", info["unicycle_pitch"])
            if "x_position" in info:
                self.logger.record("student/x_position", info["x_position"])
            if "y_position" in info:
                self.logger.record("student/y_position", info["y_position"])
            if "num_success" in info:
                self.logger.record("student/num_success", info["num_success"])
            if "reward" in info:
                self.logger.record("student/reward", info["reward"])

            if "student_action" in info:
                action_names = ["wheel_motor", "roll_control", "pitch_control", "yaw_control"]
                for i, val in enumerate(info["student_action"]):
                    self.logger.record(f"student/{action_names[i]}", val)
        return True

##########################################################################################################################
# 4) Coach Monitor Callback
##########################################################################################################################
class CoachMonitor(BaseCallback):
    """
    Logs coach actions, etc. (refer to info["coach_action"]).
    """
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        action_names = ["wheel_motor", "roll_control", "pitch_control", "yaw_control"]
        for info in infos:
            if "coach_action" in info:
                coach_action = info["coach_action"]
                # If the coach action is multi-dimensional, log each component
                for i, val in enumerate(coach_action):
                    self.logger.record(f"coach/{action_names[i]}", val)
            if "coach_action_prime" in info:
                coach_action_prime = info["coach_action_prime"]
                for i, val in enumerate(coach_action_prime):
                    self.logger.record(f"coach/prime_{action_names[i]}", val)
        return True


##########################################################################################################################
# 5) Train Function
##########################################################################################################################
def train(env, total_timesteps=1_000_000):
    # Create the SAC model
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        device='cuda',
        tensorboard_log=logs_dir
    )

    # Evaluation environment & callback
    eval_env = gym.make('Unicycle-v0')
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=logs_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Monitor callbacks
    student_monitor = StudentMonitor()
    coach_monitor = CoachMonitor()

    callback_list = [eval_callback, student_monitor, coach_monitor]

    # Start training
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list
    )

    final_model_path = os.path.join(final_model_dir, f"unicycle_coach_student_fixed_alpha_{timestamp}_final")
    model.save(final_model_path)
    print(f"[Train] Finished. Final model saved at: {final_model_path}")

##########################################################################################################################
# 6) Test Function
##########################################################################################################################
def test(env, path_to_model):
    model = SAC.load(path_to_model)
    print(f"[Test] Loaded model from: {path_to_model}")

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

            # Optionally render
            if hasattr(env, "render_mode") and env.render_mode == 'human':
                env.render()

    except KeyboardInterrupt:
        print("[Test] Interrupted (Ctrl + C).")
    finally:
        env.close()

##########################################################################################################################
# 7) Main Execution
##########################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coach + Student (Fixed Alpha) Training/Testing')
    parser.add_argument('-train', '--train', action='store_true', help='Train the model.')
    parser.add_argument('-test', '--test', type=str, help='Test the model (path to model file).')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Number of timesteps for training.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Fixed alpha for coach-student action merging.')
    parser.add_argument('--include_coach_in_obs', action='store_true',
                        help='If set, concat coach action to the student observation.')
    args = parser.parse_args()

    # Fixed coach model filename
    fixed_coach_file = "unicycle_trained.zip"
    coach_model_path = os.path.join(trained_models_dir, fixed_coach_file)

    # Check if the coach model file exists
    if not os.path.isfile(coach_model_path):
        raise FileNotFoundError(f"Coach model file not found: {coach_model_path}")

    # Set random seed (example)
    seed = int(time.time())
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Train mode
    if args.train:
        base_env = gym.make('Unicycle-v0')

        # Wrap environment with a fixed alpha coach-student merging
        wrapped_env = CoachStudentFixedAlphaWrapper(
            env=base_env,
            coach_model_path=coach_model_path,
            alpha=args.alpha,
            include_coach_in_obs=args.include_coach_in_obs
        )

        train(env=wrapped_env, total_timesteps=args.timesteps)
        wrapped_env.close()

    # Test mode
    if args.test:
        student_model_path = args.test
        if not os.path.isfile(student_model_path):
            print(f"[Test] Model file not found: {student_model_path}")
        else:
            test_env = gym.make('Unicycle-v0', render_mode='human')
            test_env = CoachStudentFixedAlphaWrapper(
                env=test_env,
                coach_model_path=coach_model_path,
                alpha=args.alpha,
                include_coach_in_obs=args.include_coach_in_obs
            )
            test(test_env, student_model_path)
            test_env.close()
