import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import random
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium import spaces

##########################################################################################################################
# Coach-Student Action Merging Wrapper (원래 외발자전거 코드의 레퍼런스 모델 경로 설정은 그대로 유지)
##########################################################################################################################
class CoachStudentHFWrapper(gym.Wrapper):
    """
    미리 학습된 코치 SAC 모델을 불러와 학생의 액션과 혼합합니다.
    
    - alpha: 코치와 학생 액션의 혼합 비율 (0이면 학생만, 1이면 코치만)
    - include_coach_in_obs: True이면 관측(observation)에 코치 액션을 추가
    """
    def __init__(self, env, reference_model_path, alpha=0.5, include_coach_in_obs=False):
        super().__init__(env)
        self.coach_model = SAC.load(reference_model_path, env=None)
        self.alpha = alpha
        self.include_coach_in_obs = include_coach_in_obs

        self.raw_obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        if self.include_coach_in_obs:
            new_obs_dim = self.raw_obs_dim + self.action_dim
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=env.observation_space.dtype)
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
            raise RuntimeError("Wrapper must be reset() before calling step().")

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
# 1) Basic Directory Setup (휴머노이드 코드와 동일)
##########################################################################################################################
current_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
data_dir = os.path.join(current_dir, "data")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"coach_humanoid_{timestamp}"

models_dir = os.path.join(data_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model", run_name)
final_model_dir = os.path.join(models_dir, "final_model")
logs_dir = os.path.join(data_dir, "logs", run_name)

directories = [data_dir, models_dir, best_model_dir, final_model_dir, logs_dir]
for d in directories:
    os.makedirs(d, exist_ok=True)

##########################################################################################################################
# 2) Environment Creation Function with Optional Coach Wrapper
##########################################################################################################################
def make_humanoid_env(
    seed=0,
    forward_reward_weight=1.25,
    ctrl_cost_weight=0.1,
    contact_cost_weight=5e-7,
    contact_cost_range=(-np.inf, 10.0),
    healthy_reward=5.0,
    terminate_when_unhealthy=True,
    healthy_z_range=(1.0, 2.0),
    reset_noise_scale=1e-2,
    exclude_current_positions_from_observation=True,
    include_cinert_in_observation=True,
    include_cvel_in_observation=True,
    include_qfrc_actuator_in_observation=True,
    include_cfrc_ext_in_observation=True,
    render_mode=None,
    use_coach=False,
    reference_model_path=None,
    coach_alpha=0.5,
    include_coach_in_obs=False
):
    def _init():
        env = gym.make(
            "Humanoid-v5",
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=contact_cost_weight,
            contact_cost_range=contact_cost_range,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            include_cinert_in_observation=include_cinert_in_observation,
            include_cvel_in_observation=include_cvel_in_observation,
            include_qfrc_actuator_in_observation=include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
            render_mode=render_mode
        )
        env = Monitor(env)
        env.reset(seed=seed)
        if use_coach:
            if reference_model_path is None:
                raise ValueError("Reference model path must be provided when using coach.")
            env = CoachStudentHFWrapper(env, reference_model_path, alpha=coach_alpha, include_coach_in_obs=include_coach_in_obs)
        return env
    return _init

##########################################################################################################################
# 3) Train Function (EvalCallback 포함)
##########################################################################################################################
def train(
    total_timesteps,
    num_envs=8,
    device="cuda",
    forward_reward_weight=1.25,
    ctrl_cost_weight=0.1,
    contact_cost_weight=5e-7,
    contact_cost_range=(-np.inf, 10.0),
    healthy_reward=5.0,
    terminate_when_unhealthy=True,
    healthy_z_range=(1.0, 2.0),
    reset_noise_scale=1e-2,
    exclude_current_positions_from_observation=True,
    include_cinert_in_observation=True,
    include_cvel_in_observation=True,
    include_qfrc_actuator_in_observation=True,
    include_cfrc_ext_in_observation=True,
    use_coach=False,
    reference_model_path=None,
    coach_alpha=0.5,
    include_coach_in_obs=False,
    base_seed=1
):
    train_env_fns = []
    for i in range(num_envs):
        train_env_fns.append(make_humanoid_env(
            seed=base_seed + i,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=contact_cost_weight,
            contact_cost_range=contact_cost_range,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            include_cinert_in_observation=include_cinert_in_observation,
            include_cvel_in_observation=include_cvel_in_observation,
            include_qfrc_actuator_in_observation=include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
            use_coach=use_coach,
            reference_model_path=reference_model_path,
            coach_alpha=coach_alpha,
            include_coach_in_obs=include_coach_in_obs
        ))
    vec_train_env = SubprocVecEnv(train_env_fns)

    eval_env_fns = [make_humanoid_env(
        seed=9999,
        forward_reward_weight=forward_reward_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        contact_cost_weight=contact_cost_weight,
        contact_cost_range=contact_cost_range,
        healthy_reward=healthy_reward,
        terminate_when_unhealthy=terminate_when_unhealthy,
        healthy_z_range=healthy_z_range,
        reset_noise_scale=reset_noise_scale,
        exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        include_cinert_in_observation=include_cinert_in_observation,
        include_cvel_in_observation=include_cvel_in_observation,
        include_qfrc_actuator_in_observation=include_qfrc_actuator_in_observation,
        include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
        use_coach=use_coach,
        reference_model_path=reference_model_path,
        coach_alpha=coach_alpha,
        include_coach_in_obs=include_coach_in_obs
    )]
    vec_eval_env = DummyVecEnv(eval_env_fns)

    model = SAC(
        'MlpPolicy',
        vec_train_env,
        verbose=1,
        device=device,
        batch_size=256,
        buffer_size=1_000_000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        tensorboard_log=logs_dir
    )

    eval_callback = EvalCallback(
        vec_eval_env,
        best_model_save_path=best_model_dir,
        log_path=logs_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="SAC_Humanoid",
        callback=eval_callback
    )

    final_model_path = os.path.join(final_model_dir, f"coach_humanoid_{timestamp}_final")
    model.save(final_model_path)
    print(f"[TRAIN] Training finished. Final model saved at: {final_model_path}")

    vec_train_env.close()
    vec_eval_env.close()

##########################################################################################################################
# 4) Test Function
##########################################################################################################################
def test(path_to_model,
         render_mode='human',
         forward_reward_weight=1.25,
         ctrl_cost_weight=0.1,
         contact_cost_weight=5e-7,
         contact_cost_range=(-np.inf, 10.0),
         healthy_reward=5.0,
         terminate_when_unhealthy=True,
         healthy_z_range=(1.0, 2.0),
         reset_noise_scale=1e-2,
         exclude_current_positions_from_observation=True,
         include_cinert_in_observation=True,
         include_cvel_in_observation=True,
         include_qfrc_actuator_in_observation=True,
         include_cfrc_ext_in_observation=True,
         use_coach=False,
         reference_model_path=None,
         coach_alpha=0.5,
         include_coach_in_obs=False):
    model = SAC.load(path_to_model)
    print(f"[TEST] Loaded model from: {path_to_model}")

    test_env = gym.make(
        "Humanoid-v5",
        render_mode=render_mode,
        forward_reward_weight=forward_reward_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        contact_cost_weight=contact_cost_weight,
        contact_cost_range=contact_cost_range,
        healthy_reward=healthy_reward,
        terminate_when_unhealthy=True,
        healthy_z_range=healthy_z_range,
        reset_noise_scale=reset_noise_scale,
        exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        include_cinert_in_observation=include_cinert_in_observation,
        include_cvel_in_observation=include_cvel_in_observation,
        include_qfrc_actuator_in_observation=include_qfrc_actuator_in_observation,
        include_cfrc_ext_in_observation=include_cfrc_ext_in_observation
    )
    test_env = Monitor(test_env)
    if use_coach:
        if reference_model_path is None:
            raise ValueError("Reference model path must be provided when using coach.")
        test_env = CoachStudentHFWrapper(test_env, reference_model_path, alpha=coach_alpha, include_coach_in_obs=include_coach_in_obs)

    obs, info = test_env.reset()
    done = False
    truncated = False

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _info = test_env.step(action)

            if done or truncated:
                obs, info = test_env.reset()
                done = False
                truncated = False

            if render_mode == 'human':
                test_env.render()

    except KeyboardInterrupt:
        print("[TEST] Test interrupted (Ctrl + C).")
    finally:
        test_env.close()

##########################################################################################################################
# 5) Main
##########################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the Humanoid SAC model with optional Coach-Student Action Merging.')
    parser.add_argument('-train', '--train', action='store_true', help='Train the model.')
    parser.add_argument('-test', '--test', type=str, help='Test the model (path to .zip file).')
    parser.add_argument('--timesteps', type=int, default=1_500_000, help='Number of timesteps for training (default 1.5M).')
    parser.add_argument('--num_envs', type=int, default=8, help='Number of parallel environments for training.')
    # 코치 머징 관련 인자
    parser.add_argument('--coach', action='store_true', help='Enable coach-student action merging.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Mixing ratio for action merging (0: student only, 0.5: mix).')
    parser.add_argument('--include_coach_in_obs', action='store_true', help='Concatenate coach action to observation if set.')
    # 시드 인자 (1~5)
    parser.add_argument('--seed', type=int, default=1, choices=range(1,6), help='Random seed (choose an integer between 1 and 5).')

    args = parser.parse_args()

    # 전역 시드 설정
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_coach = args.coach
    # 원래 외발자전거 코드에 있던 고정 레퍼런스 모델 경로
    fixed_reference_model_path = os.path.join("models", "humanoid_coach_reference_model")
    reference_model_path = fixed_reference_model_path
    coach_alpha = args.alpha
    include_coach_in_obs = args.include_coach_in_obs

    if args.train:
        train(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            device="cuda",
            forward_reward_weight=1.25,
            ctrl_cost_weight=0.1,
            contact_cost_weight=5e-7,
            contact_cost_range=(-np.inf, 10.0),
            healthy_reward=5.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(1.0, 2.0),
            reset_noise_scale=1e-2,
            exclude_current_positions_from_observation=True,
            include_cinert_in_observation=True,
            include_cvel_in_observation=True,
            include_qfrc_actuator_in_observation=True,
            include_cfrc_ext_in_observation=True,
            use_coach=use_coach,
            reference_model_path=reference_model_path,
            coach_alpha=coach_alpha,
            include_coach_in_obs=include_coach_in_obs,
            base_seed=seed
        )

    if args.test:
        test(
            path_to_model=args.test,
            render_mode='human',
            forward_reward_weight=1.25,
            ctrl_cost_weight=0.1,
            contact_cost_weight=5e-7,
            contact_cost_range=(-np.inf, 10.0),
            healthy_reward=5.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(1.0, 2.0),
            reset_noise_scale=1e-2,
            exclude_current_positions_from_observation=True,
            include_cinert_in_observation=True,
            include_cvel_in_observation=True,
            include_qfrc_actuator_in_observation=True,
            include_cfrc_ext_in_observation=True,
            use_coach=use_coach,
            reference_model_path=reference_model_path,
            coach_alpha=coach_alpha,
            include_coach_in_obs=include_coach_in_obs
        )
