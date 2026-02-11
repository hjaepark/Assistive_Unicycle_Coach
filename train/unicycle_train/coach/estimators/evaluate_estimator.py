import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

# ==========================================
# [수정 1] 같은 폴더에 있는 파일 import
# ==========================================
# 현재 폴더 경로를 시스템 경로에 추가 (안전장치)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 같은 폴더의 rdbf_estimator.py에서 클래스 가져오기
try:
    from rdbf_estimator import RDBFEstimator
except ImportError:
    # 혹시 상위 폴더에서 실행할 경우를 대비
    from coach.estimators.rdbf_estimator import RDBFEstimator

def run_evaluation():
    # ==========================================
    # [수정 2] 파일 경로 설정 (상대 경로 주의!)
    # ==========================================
    # 현재 파일 위치: .../coach/estimators/
    # 모델/데이터 위치가 프로젝트 루트에 있다고 가정하면 ../../../../ 등으로 올라가야 함
    # *연구자님 컴퓨터 환경에 맞춰서 절대 경로를 쓰는 게 제일 속 편합니다.*
    
    # 예시: 절대 경로 사용 권장
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 위치
    # 예: BASE_DIR에서 4단계 상위로 가서 models를 찾음 (구조에 따라 수정 필요)
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../../..")) 
    
    MODEL_PATH = os.path.join(PROJECT_ROOT, "train/unicycle_train/student/logs_student/run_20260204_180101_2239/models/best_student_model.zip")
    DATA_PATH = os.path.join(PROJECT_ROOT, "data/P07/P07_training_trajectory.csv")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 경로 확인용 출력
    print(f"Project Root estimated as: {PROJECT_ROOT}")
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return
    if not os.path.exists(DATA_PATH):
        print(f"[Error] 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        return

    # ==========================================
    # 2. 모델 & 데이터 로드
    # ==========================================
    print(f"Loading model...")
    model = SAC.load(MODEL_PATH, device=DEVICE, custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0})
    
    print(f"Loading user data...")
    df = pd.read_csv(DATA_PATH)
    
    # 데이터 전처리
    obs_cols = [f'obs_{i}' for i in range(19)] 
    act_cols = ['human_action_0', 'human_action_1']
    
    observations = torch.FloatTensor(df[obs_cols].values).to(DEVICE)
    human_actions = torch.FloatTensor(df[act_cols].values).to(DEVICE)
    
    # ==========================================
    # 3. 에스티메이터 초기화
    # ==========================================
    print("Initializing RDBF Estimator...")
    estimator = RDBFEstimator(
        action_dim=2,
        action_min=torch.tensor([-1.0, -1.0]),
        action_max=torch.tensor([1.0, 1.0]),
        beta=0.5,
        device=DEVICE
    )
    
    # ==========================================
    # 4. 루프 실행
    # ==========================================
    log_probs = {'Novice': [], 'Intermediate': [], 'Expert': []}
    skill_indices = []
    
    print("Running estimation loop...")
    
    with torch.no_grad():
        for t in range(len(observations)):
            obs = observations[t].unsqueeze(0)
            action = human_actions[t].unsqueeze(0)
            
            # RL 모델에서 분포(Distribution) 추출
            # (SB3 버전에 따라 get_distribution이 안 될 경우 아래 주석 해제)
            # features = model.policy.extract_features(obs)
            # latent_pi = model.policy.mlp_extractor.forward_actor(features)
            # mean_actions = model.policy.action_net(latent_pi)
            
            mean_actions, log_std, kwargs = model.policy.actor.get_action_dist_params(obs)
            expert_dist = model.policy.actor.action_dist.proba_distribution(mean_actions, log_std, **kwargs)
            
            # 에스티메이터 업데이트
            belief, skill_idx = estimator.update(action, expert_dist)
            
            log_probs['Novice'].append(belief[0].item())
            log_probs['Intermediate'].append(belief[1].item())
            log_probs['Expert'].append(belief[2].item())
            skill_indices.append(skill_idx)

    # ==========================================
    # 5. 결과 시각화
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    timesteps = range(len(observations))
    plt.stackplot(timesteps, 
                  log_probs['Novice'], 
                  log_probs['Intermediate'], 
                  log_probs['Expert'],
                  labels=['Novice', 'Intermediate', 'Expert'],
                  colors=['#ff9999', '#66b3ff', '#99ff99'], 
                  alpha=0.8)
    
    plt.title("Offline Skill Estimation Validation (User P07)", fontsize=15)
    plt.xlabel("Time Step")
    plt.ylabel("Skill Belief")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 같은 폴더에 결과 이미지 저장
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p07_estimation_result.png")
    plt.savefig(save_path)
    print(f"Estimation Complete! Check the graph at: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_evaluation()