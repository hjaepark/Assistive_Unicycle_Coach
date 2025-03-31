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
from stable_baselines3.common.monitor import Monitor
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

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"unicycle_coach_student_fixed_alpha_{timestamp}"

models_dir = os.path.join(data_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model", run_name)
final_model_dir = os.path.join(models_dir, "final_model")
logs_dir = os.path.join(data_dir, "logs", run_name)

for d in [data_dir, models_dir, best_model_dir, final_model_dir, logs_dir]:
    os.makedirs(d, exist_ok=True)

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
trained_models_dir = os.path.join(project_root, 'trained_models')

##########################################################################################################################
# 2) Coach + Student Action Merging Wrapper
##########################################################################################################################
class CoachStudentFixedAlphaWrapper(gym.Wrapper):
    """
    Loads a pre-trained coach SAC policy (file) and merges the student's action
    with the coach's action by a possibly adjustable alpha value.
    """
    def __init__(self, env, coach_model_path, alpha=0.5, include_coach_in_obs=False):
        super().__init__(env)
        self.coach_model = SAC.load(coach_model_path, env=None)
        self.alpha = alpha
        self.include_coach_in_obs = include_coach_in_obs

        self.raw_obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        if self.include_coach_in_obs:
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

    def set_alpha(self, new_alpha: float):
        self.alpha = new_alpha

    def reset(self, **kwargs):
        obs_raw, info = self.env.reset(**kwargs)
        self._last_obs_raw = obs_raw

        coach_action, _ = self.coach_model.predict(obs_raw, deterministic=True)
        if self.include_coach_in_obs:
            obs = np.concatenate([obs_raw, coach_action], axis=0)
        else:
            obs = obs_raw

        return obs, info

    def step(self, student_action):
        if self._last_obs_raw is None:
            raise RuntimeError("Wrapper must be reset() before calling step().")

        coach_action, _ = self.coach_model.predict(self._last_obs_raw, deterministic=True)
        coach_action_prime = self.alpha * (coach_action - student_action)

        final_action = student_action + coach_action_prime
        final_action = np.clip(final_action, -1.0, 1.0)

        next_obs_raw, reward, done, truncated, info = self.env.step(final_action)
        info["student_action"] = student_action
        info["coach_action_prime"] = coach_action_prime
        info["coach_action"] = coach_action

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
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]  
        for info in infos:
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
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        action_names = ["wheel_motor", "roll_control", "pitch_control", "yaw_control"]
        for info in infos:
            if "coach_action" in info:
                coach_action = info["coach_action"]
                for i, val in enumerate(coach_action):
                    self.logger.record(f"coach/{action_names[i]}", val)
            if "coach_action_prime" in info:
                coach_action_prime = info["coach_action_prime"]
                for i, val in enumerate(coach_action_prime):
                    self.logger.record(f"coach/prime_{action_names[i]}", val)
        return True

##########################################################################################################################
# 5) AlphaScheduleCallback (New)
##########################################################################################################################
class AlphaScheduleCallback(BaseCallback):
    """
    A callback that changes alpha to 0 at a specified fraction of total training steps.
    e.g. if fraction=0.75 -> after 75% of total timesteps, alpha=0
    """
    def __init__(self, fraction=0.50, verbose=0):
        super().__init__(verbose)
        self.fraction = fraction
        self.triggered = False  # To ensure we only do this once

    def _on_step(self) -> bool:
        # total_timesteps: The total number of timesteps we want to train (from model or outside)
        # num_timesteps: current timesteps so far
        if not self.triggered:
            current_steps = self.model.num_timesteps
            total_steps = self.model._total_timesteps  # the scheduled total timesteps
            if total_steps is None:  # handle case if not set
                return True

            threshold = int(total_steps * self.fraction)
            if current_steps >= threshold:
                if self.verbose > 0:
                    print(f"[AlphaScheduleCallback] Reached {threshold} steps. Setting alpha=0.")
                # set alpha=0 for train_env
                if hasattr(self.training_env, "envs"):
                    # VecEnv
                    for env_inst in self.training_env.envs:
                        if hasattr(env_inst, "set_alpha"):
                            env_inst.set_alpha(0.0)
                else:
                    if hasattr(self.training_env, "set_alpha"):
                        self.training_env.set_alpha(0.0)

                # set alpha=0 for eval_env if needed
                if "eval_callback" in self.globals:
                    eval_callback = self.globals["eval_callback"]
                    if hasattr(eval_callback.model.get_env(), "envs"):
                        for env_inst in eval_callback.model.get_env().envs:
                            if hasattr(env_inst, "set_alpha"):
                                env_inst.set_alpha(0.0)
                    else:
                        # single env
                        if hasattr(eval_callback.model.get_env(), "set_alpha"):
                            eval_callback.model.get_env().set_alpha(0.0)

                self.triggered = True
        return True

##########################################################################################################################
# 6) Train Function (Single learn call + AlphaScheduleCallback)
##########################################################################################################################
def train(env,
          total_timesteps=1_000_000,
          coach_model_path=None,
          alpha=0.5,
          alpha_fraction=0.50,  # fraction where alpha->0
          include_coach_in_obs=False):

    # create eval env
    eval_base_env = gym.make('Unicycle-v0')
    eval_base_env = Monitor(eval_base_env)
    eval_env = CoachStudentFixedAlphaWrapper(
        env=eval_base_env,
        coach_model_path=coach_model_path,
        alpha=alpha,
        include_coach_in_obs=include_coach_in_obs
    )

    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        device='cuda',
        tensorboard_log=logs_dir
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

    student_monitor = StudentMonitor()
    coach_monitor = CoachMonitor()

    alpha_schedule_callback = AlphaScheduleCallback(
        fraction=alpha_fraction,
        verbose=1
    )
    # eval_callback 접근이 필요하면 callback간 정보 공유를 세팅
    alpha_schedule_callback.globals = {"eval_callback": eval_callback}

    callback_list = [eval_callback, student_monitor, coach_monitor, alpha_schedule_callback]

    # single call -> model.learn
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list
    )

    final_model_path = os.path.join(final_model_dir,
                                    f"unicycle_coach_student_fixed_alpha_{timestamp}_final")
    model.save(final_model_path)
    print(f"[Train] Finished. Final model saved at: {final_model_path}")

##########################################################################################################################
# 7) Test Function
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

            if hasattr(env, "render_mode") and env.render_mode == 'human':
                env.render()
    except KeyboardInterrupt:
        print("[Test] Interrupted (Ctrl + C).")
    finally:
        env.close()

##########################################################################################################################
# 8) Main
##########################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coach + Student Single-run with Alpha Scheduling')
    parser.add_argument('-train', '--train', action='store_true', help='Train the model.')
    parser.add_argument('-test', '--test', type=str, help='Test the model (path to model file).')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Total timesteps for training.')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Initial alpha for the first portion of training.')
    parser.add_argument('--alpha_fraction', type=float, default=0.50,
                        help='Fraction of total steps where alpha switches to 0.')
    parser.add_argument('--no_coach_in_obs',
                        action='store_false',
                        dest='include_coach_in_obs',
                        default=True,
                        help='If set, do NOT include coach action in observation.')
    args = parser.parse_args()

    fixed_coach_file = "unicycle_trained.zip"
    coach_model_path = os.path.join(trained_models_dir, fixed_coach_file)
    if not os.path.isfile(coach_model_path):
        raise FileNotFoundError(f"Coach model file not found: {coach_model_path}")
    
    # Time based random seed
    #seed = int(time.time())
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.train:
        base_env = gym.make('Unicycle-v0')
        base_env = Monitor(base_env)
        wrapped_env = CoachStudentFixedAlphaWrapper(
            env=base_env,
            coach_model_path=coach_model_path,
            alpha=args.alpha,
            include_coach_in_obs=args.include_coach_in_obs
        )

        train(
            env=wrapped_env,
            total_timesteps=args.timesteps,
            coach_model_path=coach_model_path,
            alpha=args.alpha,
            alpha_fraction=args.alpha_fraction,
            include_coach_in_obs=args.include_coach_in_obs
        )
        wrapped_env.close()

    if args.test:
        test_model_path = args.test
        if not os.path.isfile(test_model_path):
            print(f"[Test] Model file not found: {test_model_path}")
        else:
            test_env = gym.make('Unicycle-v0', render_mode='human')
            test_env = Monitor(test_env)
            test_env = CoachStudentFixedAlphaWrapper(
                env=test_env,
                coach_model_path=coach_model_path,
                alpha=args.alpha,
                include_coach_in_obs=args.include_coach_in_obs
            )
            test(test_env, test_model_path)
            test_env.close()
