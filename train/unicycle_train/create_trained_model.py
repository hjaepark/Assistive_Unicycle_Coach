# train/unicycle_train/create_trained_model.py
# RLlib 2.12.0 – legacy (“old”) trainer stack
import os, ray, datetime
from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import Episode

# ──────────────────────────────────────
# 1)  Environment creator
# ──────────────────────────────────────
def unicycle_env_creator(env_config):
    import gymnasium as gym, envs.unicycle   # registers "Unicycle-v0"
    return gym.make("Unicycle-v0", **env_config)

tune.register_env("UnicycleRLlib", unicycle_env_creator)

# ──────────────────────────────────────
# 2)  Stop-on-success callback
# ──────────────────────────────────────
TARGET_SUCCESS = 1000
class StopOnSuccessCallback(DefaultCallbacks):
    def on_episode_end(self, *, episode: Episode, **_):
        info = episode.last_info_for()
        if info:
            episode.custom_metrics["num_success"] = info.get("num_success", 0)

    def on_train_result(self, *, result: dict, **_):
        if result["custom_metrics"].get("num_success_max", 0) >= TARGET_SUCCESS:
            print(f"[Stop] {TARGET_SUCCESS} successes reached → terminating.")
            result["done"] = True

# ──────────────────────────────────────
# 3)  SACConfig  (SB3-like 세팅)
# ──────────────────────────────────────
base_cfg = (
    SACConfig()
        .environment("UnicycleRLlib")
        .framework("torch")

        # ▶ env-step ↔ grad-step 1:1 로 맞춤
        .rollouts(num_rollout_workers=0, rollout_fragment_length=1)

        # ▶ SB3 와 동일하게 추가 노이즈 X (π 자체 σ 로만 탐색)
        #   → exploration() 호출을 생략하면 기본 StochasticSampling 사용
        .resources(num_gpus=1)

        .training(
            lr=3e-4,                   # actor LR (global)
            train_batch_size=256,
            tau=0.005,                 # SB3 기본과 동일
            target_entropy=-6.0,
            training_intensity=1.0,    # replay ratio 1:1
        )

        # ▶ 버퍼 충분히 채우고 시작
        .update_from_dict({
            "num_steps_sampled_before_learning_starts": 100
        })

        .debugging(seed=42)
        .callbacks(StopOnSuccessCallback)
)

# ▶ critic / entropy LR
config = base_cfg.update_from_dict({
    "optimization": {
        "actor_learning_rate":   3e-4,
        "critic_learning_rate":  8e-4,
        "entropy_learning_rate": 2.5e-4,
    }
})

# ──────────────────────────────────────
# 4)  Launch training
# ──────────────────────────────────────
def main():
    logs_root = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_root, exist_ok=True)

    run_name = f"unicycle_student_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    tuner = tune.Tuner(
        "SAC",
        param_space=config.to_dict(),
        run_config=RunConfig(
            name=run_name,
            storage_path=logs_root,           # logs/run_name/…
            stop={"timesteps_total": 10**7},
            checkpoint_config=CheckpointConfig(
                num_to_keep           = 3,
                checkpoint_frequency = 1,     # 매 iter
                checkpoint_at_end    = True,
                checkpoint_score_attribute = "episode_reward_mean",
                checkpoint_score_order     = "max",
            ),
        ),
    )

    results = tuner.fit()

    # 경로 출력
    best = results.get_best_result()
    if best:
        print("Best checkpoint →", best.checkpoint.path)
        # 파이널(종료시점) 체크포인트 찾기
        end_ckpt = None
        for stats in best.metrics.get("checkpoint_stats", {}).values():
            if stats.get("type") == "end_of_training":
                end_ckpt = stats["filepath"]
                break
        if end_ckpt:
            print("Final checkpoint →", end_ckpt)

if __name__ == "__main__":
    main()
