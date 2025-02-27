import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import random
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

##########################################################################################################################
##########################################################################################################################
#################### SAC + Humanoid-v5 environment training code, matching the structure of student_hf_train.py ##########
##########################################################################################################################
##########################################################################################################################

# 1) Basic directory setup (same style as student_hf_train.py)
#    - We assume this file is nested 3 levels under the project root in your actual setup;
#      adjust if needed.
current_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
data_dir = os.path.join(current_dir, "data")

# Timestamp & run name
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"humanoid_{timestamp}"

# Detailed structure for models and logs
models_dir = os.path.join(data_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model", run_name)  # if needed
final_model_dir = os.path.join(models_dir, "final_model")
logs_dir = os.path.join(data_dir, "logs", run_name)

# Create directories if not exist
directories = [data_dir, models_dir, best_model_dir, final_model_dir, logs_dir]
for d in directories:
    os.makedirs(d, exist_ok=True)


##########################################################################################################################
# 2) SubprocVecEnv를 사용하기 위한 환경 생성 함수
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
    render_mode=None
):
    """
    Gymnasium의 Humanoid-v5 환경을 반환하는 클로저를 만들어주는 함수.
    SubprocVecEnv에서 각 프로세스마다 이 함수를 호출해 환경을 생성.
    (자전거 코드 대비, MuJoCo Humanoid 환경 사용)
    """
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
        # Monitor 래퍼 -> 에피소드별 reward, length 등을 기록 (SB3가 텐서보드로 표현 가능)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


##########################################################################################################################
# 3) Train function
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
    include_cfrc_ext_in_observation=True
):
    """
    병렬 환경(SubprocVecEnv)으로 Humanoid-v5를 구성하고 SAC 모델을 학습.
    (TensorBoard 로깅 포함)
    """

    # 3.1 SubprocVecEnv 병렬 환경 생성
    env_fns = []
    for i in range(num_envs):
        env_fns.append(make_humanoid_env(
            seed=i,
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
            include_cfrc_ext_in_observation=include_cfrc_ext_in_observation
        ))
    vec_env = SubprocVecEnv(env_fns)

    # 3.2 모델 생성 (TensorBoard 로그 디렉토리 지정)
    from stable_baselines3 import SAC
    model = SAC(
        'MlpPolicy',
        vec_env,
        verbose=1,
        device=device,
        batch_size=256,
        buffer_size=1_000_000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        tensorboard_log=logs_dir  # <-- 텐서보드 로그 경로 지정
    )

    # 3.3 모델 학습 (tb_log_name으로 텐서보드 내 다른 실험과 구분)
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="SAC_Humanoid"
    )

    # 3.4 모델 최종 저장 (자전거 코드 방식)
    final_model_path = os.path.join(final_model_dir, f"humanoid_{timestamp}_final")
    model.save(final_model_path)
    print(f"[TRAIN] Training finished. Final model saved at: {final_model_path}")

    # 벡터화 환경 정리
    vec_env.close()


##########################################################################################################################
# 4) Test function
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
         include_cfrc_ext_in_observation=True):
    """
    학습된 SAC 모델(.zip)을 불러와,
    Humanoid-v5 환경(단일)에서 무한 루프 테스트 (자전거 코드와 같은 구조).
    Ctrl + C 로 중단.
    """
    # 4.1 모델 로드
    model = SAC.load(path_to_model)
    print(f"[TEST] Loaded model from: {path_to_model}")

    # 4.2 테스트용 단일 환경 생성
    test_env = gym.make(
        "Humanoid-v5",
        render_mode=render_mode,
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
        include_cfrc_ext_in_observation=include_cfrc_ext_in_observation
    )

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
# 5) Main execution (아규먼트 파싱, 자전거 코드 구조와 유사)
##########################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the Humanoid SAC model.')
    parser.add_argument('-train', '--train', action='store_true', help='Train the model.')
    parser.add_argument('-test', '--test', type=str, help='Test the model (path to model file).')
    parser.add_argument('--timesteps', type=int, default=10**6, help='Number of timesteps for training.')
    parser.add_argument('--num_envs', type=int, default=8, help='Number of parallel environments for training.')
    args = parser.parse_args()

    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Train mode
    if args.train:
        train(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            device="cuda",  # GPU (필요 시 'cpu'로 변경)
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
            include_cfrc_ext_in_observation=True
        )

    # Test mode
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
            include_cfrc_ext_in_observation=True
        )
