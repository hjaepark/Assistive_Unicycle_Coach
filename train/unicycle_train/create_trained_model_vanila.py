# train/unicycle_train/create_trained_model.py
# RLlib 2.12.0 – legacy(“old”) stack  |  *vanilla* SAC hyper-params

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
    import gymnasium as gym, envs.unicycle     # registers "Unicycle-v0"
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
# 3)  *Vanilla* SACConfig
#     - 아무 파라미터도 손대지 않음
#     - env / framework / GPU / 콜백만 명시
# ──────────────────────────────────────
config = (
    SACConfig()
        .environment("UnicycleRLlib")
        .framework("torch")
        .resources(num_gpus=1)          # GPU 사용 (기본값은 0)
        .debugging(seed=42)
        .callbacks(StopOnSuccessCallback)
).to_dict()                             # 그대로 dict 로 변환

# ──────────────────────────────────────
# 4)  Launch training
# ──────────────────────────────────────
def main():
    logs_root = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_root, exist_ok=True)

    run_name = f"unicycle_vanilla_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    tuner = tune.Tuner(
        "SAC",
        param_space=config,
        run_config=RunConfig(
            name=run_name,
            storage_path=logs_root,
            stop={"timesteps_total": 10**7},
            checkpoint_config=CheckpointConfig(
                num_to_keep            = 3,
                checkpoint_frequency   = 1,
                checkpoint_at_end      = True,
                checkpoint_score_attribute = "episode_reward_mean",
                checkpoint_score_order     = "max",
            ),
        ),
    )

    results = tuner.fit()

    best = results.get_best_result()
    if best:
        print("Best checkpoint →", best.checkpoint.path)
        # 엔드-오브-트레이닝 체크포인트
        end_ckpt = next(
            (
                st["filepath"]
                for st in best.metrics.get("checkpoint_stats", {}).values()
                if st.get("type") == "end_of_training"
            ),
            None,
        )
        if end_ckpt:
            print("Final checkpoint →", end_ckpt)

if __name__ == "__main__":
    main()
