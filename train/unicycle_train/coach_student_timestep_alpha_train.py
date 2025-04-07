# coach_student_timestep_alpha_train.py

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
run_name = f"unicycle_coach_student_timestep_alpha_{timestamp}"

models_dir = os.path.join(data_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model", run_name)
final_model_dir = os.path.join(data_dir, "final_model")
logs_dir = os.path.join(data_dir, "logs", run_name)

for d in [data_dir, models_dir, best_model_dir, final_model_dir, logs_dir]:
    os.makedirs(d, exist_ok=True)

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
trained_models_dir = os.path.join(project_root, 'trained_models')

##########################################################################################################################
# 2) Coach + Student Action Merging Wrapper
##########################################################################################################################
class CoachStudentTimestepAlphaWrapper(gym.Wrapper):
    """
    Loads a pre-trained coach SAC policy (file) and merges the student's action
    with the coach's action by a possibly adjustable alpha value.
    """
    def __init__(self, env, coach_model_path, alpha=1.0, include_coach_in_obs=False):
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
    """
    Custom callback to record student-specific observations and actions.
    """
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
    """
    Custom callback to record coach-specific actions.
    """
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
# 5) TimestepAlphaCallback (Linear decay from alpha_init to alpha_final)
##########################################################################################################################
class TimestepAlphaCallback(BaseCallback):
    """
    Linearly decays alpha from alpha_init down to alpha_final over a fraction of total steps.
    After that fraction, alpha remains alpha_final.
    """
    def __init__(self, alpha_init=1.0, alpha_final=0.0, fraction=0.5, verbose=0):
        super().__init__(verbose)
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.fraction = fraction

    def _on_training_start(self) -> None:
        # At training start, set alpha to alpha_init
        if hasattr(self.training_env, "envs"):
            for e in self.training_env.envs:
                if hasattr(e, "set_alpha"):
                    e.set_alpha(self.alpha_init)
        else:
            if hasattr(self.training_env, "set_alpha"):
                self.training_env.set_alpha(self.alpha_init)

    def _on_step(self) -> bool:
        current_step = self.model.num_timesteps
        total_steps = self.model._total_timesteps

        if total_steps is None:
            return True

        # The number of steps over which alpha decays linearly
        decay_steps = int(total_steps * self.fraction)

        if current_step <= decay_steps:
            ratio = current_step / float(decay_steps)  # from 0 to 1
            new_alpha = self.alpha_init + ratio * (self.alpha_final - self.alpha_init)
        else:
            # after decay_steps, keep alpha_final
            new_alpha = self.alpha_final

        # Update alpha
        if hasattr(self.training_env, "envs"):
            for e in self.training_env.envs:
                if hasattr(e, "set_alpha"):
                    e.set_alpha(new_alpha)
        else:
            if hasattr(self.training_env, "set_alpha"):
                self.training_env.set_alpha(new_alpha)

        # Record alpha for TensorBoard
        self.logger.record("alpha/current_alpha", new_alpha)

        return True

##########################################################################################################################
# 6) Train Function (Single learn call)
##########################################################################################################################
def train(env,
          total_timesteps=1_000_000,
          coach_model_path=None,
          alpha_init=1.0,
          alpha_final=0.0,
          alpha_fraction=0.5,
          include_coach_in_obs=False):
    """
    Train the SAC model by wrapping the environment with 
    CoachStudentTimestepAlphaWrapper, 
    then linearly decaying alpha from alpha_init to alpha_final 
    over (fraction * total_timesteps).
    """
    # Create eval env (same wrapper but for evaluation)
    eval_base_env = gym.make('Unicycle-v0')
    eval_base_env = Monitor(eval_base_env)
    eval_env = CoachStudentTimestepAlphaWrapper(
        env=eval_base_env,
        coach_model_path=coach_model_path,
        alpha=alpha_init,  # start from alpha_init
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

    # The linear alpha decay callback
    alpha_callback = TimestepAlphaCallback(
        alpha_init=alpha_init,
        alpha_final=alpha_final,
        fraction=alpha_fraction,
        verbose=1
    )

    callback_list = [eval_callback, student_monitor, coach_monitor, alpha_callback]

    # Single call -> model.learn
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list
    )

    final_model_path = os.path.join(final_model_dir,
                                    f"unicycle_coach_student_timestep_alpha_{timestamp}_final")
    model.save(final_model_path)
    print(f"[Train] Finished. Final model saved at: {final_model_path}")

##########################################################################################################################
# 7) Test Function
##########################################################################################################################
def test(env, path_to_model):
    """
    Test the trained model in the provided environment.
    """
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
    parser = argparse.ArgumentParser(description='Coach + Student Single-run with Timestep-based Alpha Decay')
    parser.add_argument('-train', '--train', action='store_true', help='Train the model.')
    parser.add_argument('-test', '--test', type=str, help='Test the model (path to model file).')

    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Total timesteps for training.')
    parser.add_argument('--alpha_init', type=float, default=1.0,
                        help='Initial alpha for linear decay.')
    parser.add_argument('--alpha_final', type=float, default=0.0,
                        help='Final alpha after fraction is reached.')
    parser.add_argument('--alpha_fraction', type=float, default=0.5,
                        help='Fraction of total steps to linearly decay from alpha_init to alpha_final.')

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

    # Fix random seed
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.train:
        base_env = gym.make('Unicycle-v0')
        base_env = Monitor(base_env)

        # Wrap environment with alpha_init
        wrapped_env = CoachStudentTimestepAlphaWrapper(
            env=base_env,
            coach_model_path=coach_model_path,
            alpha=args.alpha_init,
            include_coach_in_obs=args.include_coach_in_obs
        )

        train(
            env=wrapped_env,
            total_timesteps=args.timesteps,
            coach_model_path=coach_model_path,
            alpha_init=args.alpha_init,
            alpha_final=args.alpha_final,
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
            test_env = CoachStudentTimestepAlphaWrapper(
                env=test_env,
                coach_model_path=coach_model_path,
                alpha=args.alpha_init,  
                include_coach_in_obs=args.include_coach_in_obs
            )
            test(test_env, test_model_path)
            test_env.close()
