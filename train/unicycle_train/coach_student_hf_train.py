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
from gymnasium import spaces

##########################################################################################################################
# 1) Basic Directory Setup
##########################################################################################################################
current_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
)
data_dir = os.path.join(current_dir, "data")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"coach_student_hf_{timestamp}"

models_dir = os.path.join(data_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model", run_name)
final_model_dir = os.path.join(data_dir, "final_model")
logs_dir = os.path.join(data_dir, "logs", run_name)

for d in [data_dir, models_dir, best_model_dir, final_model_dir, logs_dir]:
    os.makedirs(d, exist_ok=True)

##########################################################################################################################
# 2) Coach + Student Action Merging Wrapper
##########################################################################################################################
class CoachStudentHFWrapper(gym.Wrapper):
    """
    Loads a pre-trained coach SAC policy and merges its action with the student's action.
    
    - alpha: mixing ratio between coach and student actions (0: student only, 1: coach only)
    - include_coach_in_obs: if True, concatenate coach action to the observation.
    """

    def __init__(self, env, reference_model_path, alpha=0.5, include_coach_in_obs=False):
        super().__init__(env)
        self.coach_model = SAC.load(reference_model_path, env=None)
        self.alpha = alpha
        self.include_coach_in_obs = include_coach_in_obs

        # Determine the observation and action dimensions from the environment.
        self.raw_obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Optionally expand the observation space to include the coach action.
        if self.include_coach_in_obs:
            new_obs_dim = self.raw_obs_dim + self.action_dim
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=env.observation_space.dtype)
        else:
            self.observation_space = env.observation_space

        self.action_space = env.action_space
        self._last_obs_raw = None
    
    def reset(self, **kwargs):
        # Reset environment and update last observation.
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

        # Merge student and coach actions using alpha.
        final_action = student_action + self.alpha * (coach_action - student_action)
        final_action = np.clip(final_action, -1.0, 1.0)

        # Apply the final action to the environment.
        next_obs_raw, reward, done, truncated, info = self.env.step(final_action)

        # Record coach action in observation.
        next_coach_action, _ = self.coach_model.predict(next_obs_raw, deterministic=True)
        if self.include_coach_in_obs:
            next_obs = np.concatenate([next_obs_raw, next_coach_action], axis=0)
        else:
            next_obs = next_obs_raw

        self._last_obs_raw = next_obs_raw
        return next_obs, reward, done, truncated, info

    def set_alpha(self, new_alpha: float):
        # Update the mixing ratio.
        self.alpha = np.clip(new_alpha, 0.0, 1.0)

##########################################################################################################################
# 3) Student Monitor Callback
##########################################################################################################################
class StudentMonitor(BaseCallback):
    """
    Callback for logging various statistics from the student agent.
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
            if "pendulum_roll" in info:
                self.logger.record("student/pendulum_roll", info["pendulum_roll"])
            if "pendulum_pitch" in info:
                self.logger.record("student/pendulum_pitch", info["pendulum_pitch"])
            if "x_position" in info:
                self.logger.record("student/x_position", info["x_position"])
            if "y_position" in info:
                self.logger.record("student/y_position", info["y_position"])
            if "goal_reached" in info:
                val = 1.0 if info["goal_reached"] else 0.0
                self.logger.record("student/goal_reached", val)
            if "steps" in info:
                self.logger.record("student/steps", info["steps"])
            if "num_steps" in info:
                self.logger.record("student/num_steps", info["num_steps"])
            if "num_success" in info:
                self.logger.record("student/num_success", info["num_success"])
            if "num_episodes" in info:
                self.logger.record("student/num_episodes", info["num_episodes"])
            if "reward" in info:
                self.logger.record("student/reward", info["reward"])
        return True

##########################################################################################################################
# 4) Adaptive Alpha Callback (without action discrepancy)
##########################################################################################################################
class AdaptiveAlphaCallback(BaseCallback):
    """
    Adjusts the alpha parameter using different strategies:
      - "fixed": keep alpha constant.
      - "timestep_based": linearly decay alpha.
      - "phase_based": manually set alpha in predefined phases.
      - "reward_based": adjust alpha based on recent rewards.
    """
    def __init__(
        self,
        alpha_method="fixed",
        alpha_init=0.5,
        alpha_final=0.0,
        alpha_decay_rate=1e-5,
        total_training_steps=1_500_000,
        reward_based_phase_step=150_000,
        alpha_increase=0.05,
        alpha_decrease=0.05,
        window_size=5,
        verbose=0
    ):
        super().__init__(verbose)
        self.alpha_method = alpha_method
        self.alpha_init = alpha_init
        self.alpha = alpha_init
        self.alpha_final = alpha_final
        self.alpha_decay_rate = alpha_decay_rate
        self.total_training_steps = total_training_steps

        # Parameters for reward-based adjustment.
        self.reward_based_phase_step = reward_based_phase_step
        self.alpha_increase = alpha_increase
        self.alpha_decrease = alpha_decrease
        self.window_size = window_size

        # Variables for tracking recent episode rewards.
        self.current_episode_reward = 0.0
        self.recent_returns = []  # Reward values for the latest episodes.
        self.prev_avg_return = None  # Previous average reward over a window.

    def _on_training_start(self) -> None:
        # Initialize alpha based on selected method.
        if self.alpha_method == "timestep_based":
            self.alpha = self.alpha_init
        elif self.alpha_method == "phase_based":
            self.alpha = 1.0
        elif self.alpha_method == "reward_based":
            self.alpha = self.alpha_init
        else:
            self.alpha = self.alpha_init

        self._apply_alpha_to_env()
        self.logger.record("alpha/current_alpha", self.alpha)

    def _on_step(self) -> bool:
        if self.alpha_method == "timestep_based":
            self._update_alpha_timestep_based()
            self._apply_alpha_to_env()
        elif self.alpha_method == "phase_based":
            self._update_alpha_phase_based()
            self._apply_alpha_to_env()
        elif self.alpha_method == "reward_based":
            self._update_alpha_reward_based()
        # For fixed method, do nothing.
        return True

    def _update_alpha_timestep_based(self):
        # Linear decay of alpha from alpha_init to alpha_final.
        current_step = self.num_timesteps
        ratio = current_step / float(self.total_training_steps)
        new_alpha = (1.0 - ratio) * self.alpha_init + ratio * self.alpha_final
        self.alpha = np.clip(new_alpha, 0.0, 1.0)

    def _update_alpha_phase_based(self):
        # Manually defined phases: 0-100k: 1.0, 100k-600k: 0.5, >600k: 0.0.
        current_step = self.num_timesteps
        if current_step < 100_000:
            self.alpha = 1.0
        elif current_step < 600_000:
            self.alpha = 0.5
        else:
            self.alpha = 0.0

    def _update_alpha_reward_based(self):
        # Adjust alpha based on recent reward trends.
        current_step = self.num_timesteps
        rewards = self.locals["rewards"]  # Array of rewards.
        dones = self.locals["dones"]      # Array of done flags.
        self.current_episode_reward += np.mean(rewards)

        if any(dones):
            # Episode ends: update recent returns.
            self.recent_returns.append(self.current_episode_reward)
            if len(self.recent_returns) > self.window_size:
                self.recent_returns.pop(0)
            self.current_episode_reward = 0.0

            if current_step < self.reward_based_phase_step:
                self.alpha = self.alpha_init
            else:
                if len(self.recent_returns) == self.window_size:
                    new_avg_return = np.mean(self.recent_returns)
                    if self.prev_avg_return is not None:
                        if new_avg_return > self.prev_avg_return:
                            self.alpha = max(0.0, self.alpha - self.alpha_decrease)
                        else:
                            self.alpha = min(1.0, self.alpha + self.alpha_increase)
                    self.prev_avg_return = new_avg_return

            self._apply_alpha_to_env()
            self.logger.record("alpha/current_alpha", self.alpha)
            self.logger.record("alpha/current_step", current_step)
            if self.prev_avg_return is not None:
                self.logger.record("alpha/prev_avg_return", self.prev_avg_return)

    def _apply_alpha_to_env(self):
        # Apply the updated alpha to all environments.
        if hasattr(self.training_env, "envs"):
            for e in self.training_env.envs:
                if hasattr(e, "set_alpha"):
                    e.set_alpha(self.alpha)
        else:
            if hasattr(self.training_env, "set_alpha"):
                self.training_env.set_alpha(self.alpha)
        self.logger.record("alpha/current_alpha", self.alpha)

##########################################################################################################################
# 5) Train Function
##########################################################################################################################
def train(
    env,
    total_timesteps,
    safe_coef,
    safe_margin,
    sensor_noise_std,
    action_delay,
    reference_model_path,
    alpha,
    alpha_method,
    alpha_init,
    alpha_final,
    alpha_decay_rate,
    include_coach_in_obs,
    reward_based_phase_step,  
    alpha_increase,          
    alpha_decrease,          
    window_size
):
    # Create the SAC model.
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        tensorboard_log=logs_dir
    )

    # Create evaluation environment.
    eval_base_env = gym.make(
        "StudentHF-v0",
        safe_coef=safe_coef,
        safe_margin=safe_margin,
        sensor_noise_std=sensor_noise_std,
        action_delay=action_delay,
    )

    eval_env = CoachStudentHFWrapper(
        env=eval_base_env,
        reference_model_path=reference_model_path,
        alpha=alpha,
        include_coach_in_obs=include_coach_in_obs
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
    student_monitor_callback = StudentMonitor()

    # Alpha adjustment callback.
    adaptive_alpha_callback = AdaptiveAlphaCallback(
        alpha_method=alpha_method,
        alpha_init=alpha_init,
        alpha_final=alpha_final,
        alpha_decay_rate=alpha_decay_rate,
        total_training_steps=total_timesteps,
        reward_based_phase_step=reward_based_phase_step,
        alpha_increase=alpha_increase,
        alpha_decrease=alpha_decrease,
        window_size=window_size
    )

    callbacks = [eval_callback, student_monitor_callback, adaptive_alpha_callback]

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )

    final_model_path = os.path.join(final_model_dir, f"coach_student_hf_{timestamp}_final")
    model.save(final_model_path)
    print(f"[Train] Finished. Final model saved at: {final_model_path}")

##########################################################################################################################
# 6) Test Function
##########################################################################################################################
def test(env, path_to_model):
    model = SAC.load(path_to_model)
    print(f"[Test] Loaded model from: {path_to_model}")

    obs = env.reset()[0]  # reset() returns (obs, info)
    done = False
    truncated = False

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs = env.reset()[0]
                done = False
                truncated = False
            # Optionally render the environment.
    except KeyboardInterrupt:
        print("[Test] Interrupted.")
    finally:
        env.close()

##########################################################################################################################
# 7) Main
##########################################################################################################################
def main():
    parser = argparse.ArgumentParser(description="Coach-Student HF Training")
    parser.add_argument("--train", action="store_true", help="Train with coach action merging.")
    parser.add_argument("--test", type=str, help="Test the student model (path).")
    parser.add_argument("--timesteps", type=int, default=1_500_000, help="Total training timesteps.")
    parser.add_argument("--safe_coef", type=float, default=0.02)
    parser.add_argument("--safe_margin", type=float, default=0.3)
    parser.add_argument("--sensor_noise_std", type=float, default=0.0)
    parser.add_argument("--action_delay", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5, help="Mixing ratio between coach and student actions.")

    # Alpha adjustment methods: fixed, timestep_based, phase_based, reward_based.
    parser.add_argument("--alpha_method",
                        type=str,
                        default="fixed",
                        choices=["fixed", "timestep_based", "phase_based", "reward_based"],
                        help="Strategy to update alpha."
    )
    # For timestep-based scheduling.
    parser.add_argument("--alpha_init", type=float, default=0.5, help="Initial alpha for timestep-based schedule.")
    parser.add_argument("--alpha_final", type=float, default=0.0, help="Final alpha for timestep-based schedule.")
    parser.add_argument("--alpha_decay_rate", type=float, default=1e-5, help="Decay rate (if applicable).")

    # For reward-based alpha adjustment.
    parser.add_argument("--reward_based_phase_step", type=int, default=150000)
    parser.add_argument("--alpha_increase", type=float, default=0.05)
    parser.add_argument("--alpha_decrease", type=float, default=0.05)
    parser.add_argument("--window_size", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--include_coach_in_obs", action="store_false",
                        help="If set, concatenate coach action to observation.")

    args = parser.parse_args()

    # Set seeds.
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    fixed_reference_model_path = os.path.join("models", "unicycle_coach_hf_reference_model")

    if args.train:
        base_env = gym.make(
            "StudentHF-v0",
            safe_coef=args.safe_coef,
            safe_margin=args.safe_margin,
            sensor_noise_std=args.sensor_noise_std,
            action_delay=args.action_delay
        )
        wrapped_env = CoachStudentHFWrapper(
            env=base_env,
            reference_model_path=fixed_reference_model_path,
            alpha=args.alpha,
            include_coach_in_obs=args.include_coach_in_obs
        )

        train(
            env=wrapped_env,
            total_timesteps=int(args.timesteps),
            safe_coef=args.safe_coef,
            safe_margin=args.safe_margin,
            sensor_noise_std=args.sensor_noise_std,
            action_delay=args.action_delay,
            reference_model_path=fixed_reference_model_path,
            alpha=args.alpha,
            include_coach_in_obs=args.include_coach_in_obs,
            alpha_method=args.alpha_method,
            alpha_init=args.alpha_init,
            alpha_final=args.alpha_final,
            alpha_decay_rate=args.alpha_decay_rate,
            reward_based_phase_step=args.reward_based_phase_step,
            alpha_increase=args.alpha_increase,
            alpha_decrease=args.alpha_decrease,
            window_size=args.window_size
        )
        wrapped_env.close()

    if args.test:
        test_model_path = args.test
        if not os.path.isfile(test_model_path):
            print(f"[Test] Model file not found: {test_model_path}")
            return

        test_env = gym.make(
            "StudentHF-v0",
            render_mode='human',
            safe_coef=args.safe_coef,
            safe_margin=args.safe_margin,
            sensor_noise_std=args.sensor_noise_std,
            action_delay=args.action_delay
        )

        test_env = CoachStudentHFWrapper(
            env=test_env,
            reference_model_path=fixed_reference_model_path,
            alpha=args.alpha,
            include_coach_in_obs=args.include_coach_in_obs
        )
        test(test_env, test_model_path)


if __name__ == "__main__":
    main()
