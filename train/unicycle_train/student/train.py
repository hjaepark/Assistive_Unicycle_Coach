# train/unicycle_train/student/train.py

import datetime
import random
from typing import Any, Dict

import ray
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.tune.stopper import FunctionStopper
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS

from train.unicycle_train import path as P

def unicycle_env_creator(env_config: Dict[str, Any] | None = None):
    import gymnasium as gym, envs.unicycle
    env_config = env_config or {}
    env = gym.make("UnicyclePendulumPD-v0", **env_config)
    return env

tune.register_env("UnicycleStudentRLlib", unicycle_env_creator)

# Training configuration
EVALUATION_INTERVAL = 50
EVALUATION_EPISODES = 100
REQUIRED_SUCCESS_RATE = 0.95
REQUIRED_CONSECUTIVE_PASSES = 2
MAXIMUM_TIMESTEPS = 10_000_000

class MetricsCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index=None,
        rl_module=None,
        **kwargs,
    ):
        last_info = None
        try:
            step_info = episode.get_infos(-1) 
        except Exception:
            step_info = None

        if isinstance(step_info, dict):
            if step_info and all(isinstance(v, dict) for v in step_info.values()):
                last_info = next(iter(step_info.values()), None)
            else:
                last_info = step_info
        elif isinstance(step_info, (list, tuple)):
            for v in reversed(step_info):
                if isinstance(v, dict):
                    last_info = v
                    break
        else:
            last_info = step_info if isinstance(step_info, dict) else None

        if not (metrics_logger and isinstance(last_info, dict)):
            return

        success = float(bool(last_info.get("goal_reached", False)))
        fall    = float(bool(last_info.get("is_fallen", False)))
        x_final = float(last_info.get("x_position", last_info.get("x_final", 0.0)))
        y_final = float(last_info.get("y_position", last_info.get("y_final", 0.0)))

        metrics_logger.log_value("success", success)
        metrics_logger.log_value("fall", fall)
        metrics_logger.log_value("x_final", x_final)
        metrics_logger.log_value("y_final", y_final)

    def on_train_result(self, *, result: dict, **kwargs):
        iteration = result.get("training_iteration", 0)
        if iteration % 5 != 0:
            return

        env_runners = result.get(ENV_RUNNER_RESULTS, {}) or {}
        def m(d, k, default=0.0):
            return d.get(k, d.get(f"{k}_mean", default))

        episode_return = float(env_runners.get("episode_return_mean", 0.0))
        total_steps = int(result.get("num_env_steps_sampled_lifetime", 0))

        success = float(m(env_runners, "success"))
        fall    = float(m(env_runners, "fall"))
        x_final = float(m(env_runners, "x_final"))
        y_final = float(m(env_runners, "y_final"))

        print(f"\n[Iter {iteration:d}]")
        print(f"  Return:   {episode_return:.1f}")
        print(f"  Success:  {success:.1%}")
        print(f"  Fall:     {fall:.1%}")
        print(f"  Position: ({x_final:.2f}, {y_final:.2f})")
        print(f"  Steps:    {total_steps:,}")

certificate_state = {
    "consecutive_passes": 0, 
    "best_success_rate": 0.0,
    "evaluation_count": 0
}

last_evaluation_iteration = {"value": None}

def stopper(_, result: Dict[str, Any]) -> bool:
    it = int(result.get("training_iteration", 0))
    evaluation_results = result.get("evaluation")
    if not evaluation_results:
        return False

    if it % EVALUATION_INTERVAL != 0:
        return False
    if last_evaluation_iteration["value"] == it:
        return False
    last_evaluation_iteration["value"] = it

    env_runners = evaluation_results.get(ENV_RUNNER_RESULTS, {}) or {}

    def get_metric(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
        v = d.get(key, d.get(f"{key}_mean", default))
        try:
            return float(v)
        except Exception:
            return default

    success_rate   = get_metric(env_runners, "success")
    fall_rate      = get_metric(env_runners, "fall")
    x_final        = get_metric(env_runners, "x_final")
    y_final        = get_metric(env_runners, "y_final")
    episode_return = float(env_runners.get("episode_return_mean",
                                           env_runners.get("episode_return_mean_mean", 0.0)))

    passed = success_rate >= REQUIRED_SUCCESS_RATE
    certificate_state["consecutive_passes"] = (certificate_state["consecutive_passes"] + 1) if passed else 0
    certificate_state["best_success_rate"]  = max(certificate_state["best_success_rate"], success_rate)
    certificate_state["evaluation_count"]  += 1

    total_steps = int(result.get("num_env_steps_sampled_lifetime", 0))

    print(f"\n[Evaluation {certificate_state['evaluation_count']}] {'PASS' if passed else 'FAIL'}")
    print(f"  Success Rate: {success_rate:.1%} (threshold: {REQUIRED_SUCCESS_RATE:.1%})")
    print(f"  Fall Rate: {fall_rate:.1%}")
    print(f"  Final Position: ({x_final:.2f}, {y_final:.2f})")
    print(f"  Episode Return: {episode_return:.1f}")
    print(f"  Consecutive Passes: {certificate_state['consecutive_passes']}/{REQUIRED_CONSECUTIVE_PASSES}")
    print(f"  Steps: {total_steps:,}/{MAXIMUM_TIMESTEPS:,}")

    if certificate_state["consecutive_passes"] >= REQUIRED_CONSECUTIVE_PASSES:
        print("\nTraining completed!")
        return True

    if total_steps >= MAXIMUM_TIMESTEPS:
        print("\nMaximum timesteps reached.")
        return True

    return False

def main():
    ray.shutdown()
    ray.init()
    
    config = (
        SACConfig()
        .environment(
            env="UnicycleStudentRLlib",
            env_config={
                "reset_noise_scale": 0.01,
                "kp": 40.0,
                "kd": 6.0
            }
        )
        .framework("torch")
        .training(
            train_batch_size=256,
            gamma=0.99,
            tau=0.005,
            initial_alpha=1.0,
            target_entropy="auto",
            n_step=1,
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": 100000,
            },
        )
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=1,
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .evaluation(
            evaluation_interval=EVALUATION_INTERVAL,
            evaluation_duration=EVALUATION_EPISODES,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=2,
            evaluation_parallel_to_training=True,
            evaluation_config={
                "explore": False,
                "env_config": {
                    "reset_noise_scale": 0.01,
                    "kp": 40.0,
                    "kd": 6.0
                },
            },
        )
        .callbacks(MetricsCallback)
        .debugging(seed=42)
        .reporting(
            keep_per_episode_custom_metrics=True,
            metrics_num_episodes_for_smoothing=100,
        )
    )
    
    tuner = tune.Tuner(
        "SAC",
        param_space=config.to_dict(),
        run_config=RunConfig(
            name=f"unicycle_{datetime.datetime.now():%Y%m%d_%H%M%S}",
            storage_path=str(P.LOGS_STUDENT_DIR),
            stop=FunctionStopper(stopper),
            callbacks=[TBXLoggerCallback()],
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=50,
                num_to_keep=2,
                checkpoint_score_attribute="env_runners/episode_return_mean",
                checkpoint_score_order="max",
            ),
        ),
    )
    
    results = tuner.fit()
    
    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean",
        mode="max"
    )
    
    if best_result and best_result.checkpoint:
        save_path = P.EXPERT_MODEL_DIR / f"best_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        save_path.mkdir(parents=True, exist_ok=True)
        best_result.checkpoint.to_directory(str(save_path))
        
        print(f"\n{'='*50}")
        print(f"[Training Completed]")
        print(f"Model saved to: {save_path}")
        print(f"Best Success Rate: {certificate_state['best_success_rate']:.1%}")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()