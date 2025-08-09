# train/unicycle_train/student/train.py


import datetime
from typing import Any, Dict

import ray
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.tune.stopper import FunctionStopper
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SACConfig

from train.unicycle_train import path as P

def unicycle_env_creator(env_config: Dict[str, Any] | None = None):
    import gymnasium as gym, envs.unicycle 
    env_config = env_config or {}
    return gym.make("UnicyclePendulumPD-v0", **env_config)

tune.register_env("UnicycleStudentRLlib", unicycle_env_creator)

# Training evaluation & early termination configuration
EVALUATION_INTERVAL = 20
EVALUATION_EPISODES = 200
REQUIRED_SUCCESSES_PER_EVALUATION = 187 # ≈ 93.5% success rate. Lower confidence bound ≥ 0.90.
REQUIRED_CONSECUTIVE_PASSES = 2
MIN_EVALUATION_EPISODES = 200
MAXIMUM_TIMESTEPS = 10_000_000 # Maximum number of total timesteps.

EVALUATION_ENV_CONFIG = {
    "reset_noise_scale": 0.01,
    "kp": 40.0,
    "kd": 6.0,
}

class MetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, episode, metrics_logger=None, env_runner=None, **__):
        info = episode.get_infos(-1) or {}
        success_flag = 1 if info.get("goal_reached", False) else 0
        x_final = float(info.get("x_position", 0.0))
        y_final = float(info.get("y_position", 0.0))
        num_success_global = int(info.get("num_success", 0))

        if metrics_logger:
            metrics_logger.log_value(("custom_metrics", "success_flag"), success_flag, reduce="mean")
            metrics_logger.log_value(("custom_metrics", "x_final"), x_final, reduce="mean")
            metrics_logger.log_value(("custom_metrics", "y_final"), y_final, reduce="mean")
            metrics_logger.log_value(("custom_metrics", "num_success_global"), num_success_global, reduce="max")

certificate_state = {"consecutive_passes": 0}

def stopper(_, result: Dict[str, Any]) -> bool:
    evaluation = result.get("evaluation", {}) or {}
    custom_metrics = evaluation.get("custom_metrics", {}) or {}
    success_mean = custom_metrics.get("success_flag_mean", None)
    num_episodes = int(evaluation.get("num_episodes", 0))

    if success_mean is not None and num_episodes >= MIN_EVALUATION_EPISODES:
        successes = int(round(success_mean * num_episodes))
        passed = successes >= REQUIRED_SUCCESSES_PER_EVALUATION
        certificate_state["consecutive_passes"] = (certificate_state["consecutive_passes"] + 1) if passed else 0
        print(f"[STOP] successes={successes}/{num_episodes}  passed={passed}  "
              f"streak={certificate_state['consecutive_passes']}/{REQUIRED_CONSECUTIVE_PASSES}")
    
    reached_max_timesteps = int(result.get("timesteps_total", 0)) >= MAXIMUM_TIMESTEPS
    return (certificate_state["consecutive_passes"] >= REQUIRED_CONSECUTIVE_PASSES) or reached_max_timesteps

stopper = FunctionStopper(stopper)

sac_cfg = (
    SACConfig()
    .environment("UnicycleStudentRLlib")
    .framework("torch")
    .resources(num_gpus=1)
    .callbacks(MetricsCallback)
    .evaluation(
        evaluation_interval=EVALUATION_INTERVAL,
        evaluation_duration=EVALUATION_EPISODES,
        evaluation_duration_unit="episodes",
        evaluation_config={"explore": False, "env_config": EVALUATION_ENV_CONFIG},
    )
).to_dict()


def main() -> None:
    ray.shutdown()
    ray.init()

    logs_root = P.LOGS_STUDENT_DIR
    run_name = f"unicycle_student_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    tuner = tune.Tuner(
        "SAC",
        param_space=sac_cfg,
        run_config=RunConfig(
            name=run_name,
            storage_path=str(logs_root),
            stop=stopper,
            callbacks=[TBXLoggerCallback()],
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=5,
                num_to_keep=3,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max",
            ),
        ),
    )
    results = tuner.fit()

    best_result   = results.get_best_result(metric="episode_reward_mean", mode="max")
    latest_result = results.get_best_result(metric="training_iteration", mode="max")

    best_checkpoint   = best_result.checkpoint
    latest_checkpoint = latest_result.checkpoint

    destination_root = P.EXPERT_MODEL_DIR
    destination_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_checkpoint.to_directory(destination_root / f"best_{stamp}")
    latest_checkpoint.to_directory(destination_root / f"latest_{stamp}")

if __name__ == "__main__":
    main()