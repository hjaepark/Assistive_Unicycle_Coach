import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import random
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


##########################################################################################################################
# 1) Coach-Student Action Merging Wrapper
##########################################################################################################################
class CoachHumanoidWrapper(gym.Wrapper):
    """
    Humanoid-v5 환경에 대해, 이미 학습된 SAC 모델(코치)의 정책을 로드하고
    학생 액션과 알파 비율로 혼합하여 최종 액션을 만드는 Wrapper.

    final_action = student_action + alpha * (coach_action - student_action)
    """

    def __init__(self, env, reference_model_path, alpha=0.5):
        super().__init__(env)
        self.alpha = alpha
        self.coach_model = SAC.load(reference_model_path)

        # 원본 환경과 동일하게 observation_space, action_space 설정
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, student_action):
        if self._last_obs is None:
            raise RuntimeError("환경이 reset되지 않았습니다.")

        # 코치 모델의 액션
        coach_action, _ = self.coach_model.predict(self._last_obs, deterministic=True)

        # 학생 액션과 코치 액션 합성
        final_action = student_action + self.alpha * (coach_action - student_action)
        final_action = np.clip(final_action, self.action_space.low, self.action_space.high)

        obs, reward, done, truncated, info = self.env.step(final_action)
        self._last_obs = obs

        return obs, reward, done, truncated, info


##########################################################################################################################
# 2) Train Function
##########################################################################################################################
def train(
    reference_model_path: str,
    alpha: float,
    seed: int,
    total_timesteps: int,
    device: str = "cuda"
):
    """
    이미 학습된 코치 모델(reference_model_path)을 사용하여
    학생 SAC 모델을 Humanoid-v5 환경에서 학습합니다.
    """
    # 폴더/파일 세팅
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"humanoid_coach_{timestamp}_alpha{alpha}_seed{seed}"
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 로그 및 모델 저장 디렉토리
    data_dir = os.path.join(current_dir, "data")
    models_dir = os.path.join(data_dir, "models")
    logs_dir = os.path.join(data_dir, "logs", run_name)
    final_model_dir = os.path.join(models_dir, "final_model")

    for d in [data_dir, models_dir, logs_dir, final_model_dir]:
        os.makedirs(d, exist_ok=True)

    # 시드 고정
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 단일 환경 생성
    # (학습된 코치 모델과 동일한 환경 설정을 맞춰주세요. 
    #  아래는 기본 Humanoid-v5 예시)
    env = gym.make("Humanoid-v5")

    # Monitor로 감싸서 학습 로그(episode_reward 등) 기록
    env = Monitor(env)

    # 코치 액션 합성 Wrapper 적용
    env = CoachHumanoidWrapper(env, reference_model_path, alpha=alpha)

    # SAC 모델 생성(학생)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log=logs_dir,
        # 필요에 따라 하이퍼파라미터 조정
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=1_000_000,
        gamma=0.99,
        tau=0.005,
    )

    # 학습
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"SAC_Humanoid_Coach_alpha{alpha}")

    # 최종 모델 저장
    final_model_path = os.path.join(final_model_dir, f"humanoid_coach_student_{timestamp}_final.zip")
    model.save(final_model_path)
    print(f"[TRAIN] Finished. Final model saved at: {final_model_path}")

    env.close()


##########################################################################################################################
# 3) Test Function
##########################################################################################################################
def test(
    reference_model_path: str,
    student_model_path: str,
    alpha: float,
    seed: int,
    render_mode: str = "human"
):
    """
    - 학습 완료된 학생 모델(student_model_path)을 로드하고,
    - 코치 모델(reference_model_path)와 액션 합성을 수행하며
      Humanoid-v5 환경을 테스트.
    """
    # 시드 고정
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # 단일 테스트 환경
    env = gym.make("Humanoid-v5", render_mode=render_mode)
    env = CoachHumanoidWrapper(env, reference_model_path, alpha=alpha)

    # 학습된 학생 모델 로드
    student_model = SAC.load(student_model_path)
    print(f"[TEST] Loaded student model from: {student_model_path}")

    obs, info = env.reset()
    done = False
    truncated = False

    try:
        while True:
            action, _ = student_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _info = env.step(action)

            if done or truncated:
                obs, info = env.reset()
                done = False
                truncated = False

            # 사람이 볼 수 있도록 render
            if render_mode == "human":
                env.render()
    except KeyboardInterrupt:
        print("[TEST] Interrupted by user.")
    finally:
        env.close()


##########################################################################################################################
# 4) Main (CLI)
##########################################################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train a student policy with action merging.")
    parser.add_argument("--test", type=str, help="Path to the trained student model (.zip) for testing.")
    parser.add_argument("--ref_model_path", type=str, required=True, help="Path to the reference coach model (.zip).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fixed alpha (0=student-only, 1=coach-only)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (1~5 권장).")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Number of timesteps for training.")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")

    args = parser.parse_args()

    if args.train:
        train(
            reference_model_path=args.ref_model_path,
            alpha=args.alpha,
            seed=args.seed,
            total_timesteps=args.timesteps,
            device=args.device
        )

    if args.test:
        # --test 인자에 전달된 student 모델 경로
        test(
            reference_model_path=args.ref_model_path,
            student_model_path=args.test,
            alpha=args.alpha,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
