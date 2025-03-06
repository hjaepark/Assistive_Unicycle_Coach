#coach_student_hf_train.py

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
# 1) Basic directory setup
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
    코치(전문가) 모델에서 SAC 정책 로드, 학생 액션과 합성 + (옵션) 코치 액션을 관측에 추가.
    
    - alpha: 코치와 학생 액션 혼합 비율 (0=학생 온리, 1=코치 온리)
    - include_coach_in_obs=True 이면, 관측 obs에 coach_action을 concat하여 반환.
      ex) obs => shape (original_obs_dim + action_dim)
    """

    def __init__(
        self,
        env,
        reference_model_path,
        alpha=0.5,
        include_coach_in_obs=False
    ):
        super().__init__(env)
        self.coach_model = SAC.load(reference_model_path, env=None)
        self.alpha = alpha
        self.include_coach_in_obs = include_coach_in_obs

        # 코치가 학습된 환경의 관측 차원을 파악(= env.observation_space.shape[0])
        self.raw_obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # (옵션) 관측 공간 확장
        if self.include_coach_in_obs:
            new_obs_dim = self.raw_obs_dim + self.action_dim
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(new_obs_dim,),
                dtype=env.observation_space.dtype
            )
        else:
            self.observation_space = env.observation_space

        self.action_space = env.action_space
        self._last_obs_raw = None
    
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
            raise RuntimeError("Wrapper env must be reset() before step().")

        coach_action, _ = self.coach_model.predict(self._last_obs_raw, deterministic=True)

        final_action = student_action + self.alpha * (coach_action - student_action)
        final_action = np.clip(final_action, -1.0, 1.0)

        next_obs_raw, reward, done, truncated, info = self.env.step(final_action)

        next_coach_action, _ = self.coach_model.predict(next_obs_raw, deterministic=True)
        if self.include_coach_in_obs:
            next_obs = np.concatenate([next_obs_raw, next_coach_action], axis=0)
        else:
            next_obs = next_obs_raw

        self._last_obs_raw = next_obs_raw
        return next_obs, reward, done, truncated, info

    def set_alpha(self, new_alpha: float):
        self.alpha = np.clip(new_alpha, 0.0, 1.0)

##########################################################################################################################
# 3) StudentMonitor Callback
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
# 4) AdaptiveAlpha Callback
##########################################################################################################################
class AdaptiveAlphaCallback(BaseCallback):
    def __init__(
        self,
        alpha_method="fixed",
        alpha_init=0.5,
        alpha_final=0.0,
        alpha_decay_rate=1e-5,
        total_training_steps=1_500_000,
        verbose=0,
        
        # reward_based parameters
        alpha_reward_ema_beta=0.95,
        alpha_reward_target=1500.0,
        alpha_reward_delta=0.001
    ):
        super().__init__(verbose)
        self.alpha_method = alpha_method
        self.alpha_init = alpha_init
        self.alpha = alpha_init
        self.alpha_final = alpha_final
        self.alpha_decay_rate = alpha_decay_rate
        self.total_training_steps = total_training_steps

        # reward_based parameters
        self.alpha_reward_ema_beta = alpha_reward_ema_beta
        self.alpha_reward_target = alpha_reward_target
        self.alpha_reward_delta = alpha_reward_delta

        # EMA tracking variable
        self.r_ema = 0.0

    def _on_training_start(self) -> None:
        if self.alpha_method in ["timestep_based", "reward_based"]:
            self.alpha = self.alpha_init
            self.r_ema = 0.0
            self._apply_alpha_to_env()
            self.logger.record("alpha/current_alpha", self.alpha)

    def _on_step(self) -> bool:
        if self.alpha_method == "timestep_based":
            self._update_alpha_timestep_based()
        elif self.alpha_method == "reward_based":
            self._update_alpha_reward_based()
        elif self.alpha_method == "fixed":
            pass

        if self.alpha_method != "fixed":
            self._apply_alpha_to_env()

        return True

    def _update_alpha_timestep_based(self):
        current_step = self.num_timesteps
        ratio = current_step / float(self.total_training_steps)
        new_alpha = (1.0 - ratio) * self.alpha_init + ratio * self.alpha_final
        self.alpha = np.clip(new_alpha, 0.0, 1.0)

    def _update_alpha_reward_based(self):
        step_rewards = self.locals["rewards"]
        avg_reward = float(np.mean(step_rewards))

        beta = self.alpha_reward_ema_beta
        self.r_ema = beta * self.r_ema + (1.0 - beta) * avg_reward

        if self.r_ema >= self.alpha_reward_target:
            self.alpha = max(0.0, self.alpha - self.alpha_reward_delta)
        else:
            self.alpha = min(self.alpha_init, self.alpha + self.alpha_reward_delta)

    def _apply_alpha_to_env(self):
        if hasattr(self.training_env, "envs"):
            for e in self.training_env.envs:
                if hasattr(e, "set_alpha"):
                    e.set_alpha(self.alpha)
        else:
            if hasattr(self.training_env, "set_alpha"):
                self.training_env.set_alpha(self.alpha)
        
        self.logger.record("alpha/current_alpha", self.alpha)

##########################################################################################################################
# 5) Train function
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
    # reward_based variable
    alpha_reward_ema_beta,
    alpha_reward_target,
    alpha_reward_delta
):
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        tensorboard_log=logs_dir
    )

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

    adaptive_alpha_callback = AdaptiveAlphaCallback(
        alpha_method=alpha_method,
        alpha_init=alpha_init,
        alpha_final=alpha_final,
        alpha_decay_rate=alpha_decay_rate,
        total_training_steps=total_timesteps,

        alpha_reward_ema_beta=alpha_reward_ema_beta,
        alpha_reward_target=alpha_reward_target,
        alpha_reward_delta=alpha_reward_delta
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
# 6) Test function
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

            # Optional: env.render() if needed
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
    parser.add_argument("--timesteps", type=int, default=1_500_000, help="Training timesteps.")
    parser.add_argument("--safe_coef", type=float, default=0.02)
    parser.add_argument("--safe_margin", type=float, default=0.3)
    parser.add_argument("--sensor_noise_std", type=float, default=0.0)
    parser.add_argument("--action_delay", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5, help="Coach action ratio 0-1 for fixed alpha.")
    parser.add_argument("--alpha_method",
                        type=str,
                        default="fixed",
                        choices=["fixed", "timestep_based", "reward_based", "action_discrepancy"],
                        help="Adaptive alpha update strategy."
    )
    # timestep_based variable
    parser.add_argument("--alpha_init", type=float, default=0.5, help="Initial alpha for the timestep_based schedule.")
    parser.add_argument("--alpha_final", type=float, default=0.0, help="Final alpha for the timestep-based schedule.")
    parser.add_argument("--alpha_decay_rate", type=float, default=1e-5, help="Decay rate for custom schedules (optional).")

    # reward_based variable
    parser.add_argument("--alpha_reward_ema_beta", type=float, default=0.95,
                        help="Decay factor (beta) for EMA-based reward tracking in 'reward_based'. e.g. 0.9, 0.95, 0.99")
    parser.add_argument("--alpha_reward_target", type=float, default=500.0,
                        help="Target reward threshold for 'reward_based'.")
    parser.add_argument("--alpha_reward_delta", type=float, default=0.01,
                        help="Step size for alpha update in 'reward_based'.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--include_coach_in_obs", action="store_false", help="If set, expand obs with coach action.")

    args = parser.parse_args()

    # Seed
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

            # reward_based
            alpha_reward_ema_beta=args.alpha_reward_ema_beta,
            alpha_reward_target=args.alpha_reward_target,
            alpha_reward_delta=args.alpha_reward_delta
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
