# train/unicycle_train/student/train.py 개선안

import datetime
from typing import Any, Dict

import ray
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.tune.stopper import FunctionStopper
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SACConfig

from train.unicycle_train import path as P

def unicycle_env_creator(env_config: Dict[str, Any] | None = None):
    import gymnasium as gym, envs.unicycle
    env_config = env_config or {}
    env = gym.make("UnicyclePendulumPD-v0", **env_config)
    return env

tune.register_env("UnicycleStudentRLlib", unicycle_env_creator)

# Training configuration
EVALUATION_INTERVAL = 20
EVALUATION_EPISODES = 200
REQUIRED_SUCCESS_RATE = 0.935  # 93.5% success rate
REQUIRED_CONSECUTIVE_PASSES = 2
MAXIMUM_TIMESTEPS = 10_000_000

class MetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, episode, env_runner=None, metrics_logger=None, **kwargs):
        if hasattr(episode, 'get_infos'):
            info = episode.get_infos(-1) or {}
        else:
            info = {}
        
        if metrics_logger and info:
            metrics_logger.log_value("success", float(info.get("goal_reached", False)))
            metrics_logger.log_value("x_final", info.get("x_position", 0.0))
            metrics_logger.log_value("y_final", info.get("y_position", 0.0))
            metrics_logger.log_value("fall", float(info.get("is_fallen", False)))

    def on_train_result(self, *, algorithm, result, **kwargs):
        iteration = result["training_iteration"]
        
        if iteration % 5 == 0:
            episode_return = result.get("env_runners", {}).get("episode_return_mean", 0.0)
            total_steps = result.get("num_env_steps_sampled_lifetime", 0)
            
            success_mean = result.get("env_runners", {}).get("success_mean", 0.0)
            fall_mean = result.get("env_runners", {}).get("fall_mean", 0.0)
            
            print(f"[Iter {iteration:3d}] "
                  f"Return: {episode_return:7.1f} | "
                  f"Success: {success_mean:5.1%} | "
                  f"Fall: {fall_mean:5.1%} | "
                  f"Steps: {total_steps:,}")

def stopper(_, result: Dict[str, Any]) -> bool:
    eval_results = result.get("evaluation", {}).get("env_runners", {})
    success_rate = eval_results.get("success_mean", 0.0)
    episode_return = eval_results.get("episode_return_mean", 0.0)
    
    if success_rate >= REQUIRED_SUCCESS_RATE:
        certificate_state["consecutive_passes"] += 1
    else:
        certificate_state["consecutive_passes"] = 0
    
    certificate_state["best_success_rate"] = max(
        certificate_state["best_success_rate"], 
        success_rate
    )
    
    total_steps = result.get("num_env_steps_sampled_lifetime", 0)
    
    print(f"[Evaluation] Success Rate: {success_rate:.1%} | "
          f"Episode Return: {episode_return:.1f} | "
          f"Consecutive Passes: {certificate_state['consecutive_passes']}/{REQUIRED_CONSECUTIVE_PASSES} | "
          f"Steps: {total_steps:,}/{MAXIMUM_TIMESTEPS:,}")
    
    return (certificate_state["consecutive_passes"] >= REQUIRED_CONSECUTIVE_PASSES or 
            total_steps >= MAXIMUM_TIMESTEPS)

certificate_state = {"consecutive_passes": 0, "best_success_rate": 0.0}

def main():
    ray.shutdown()
    ray.init()
    
    config = (
        SACConfig()
        .environment(
            env="UnicycleStudentRLlib",
            env_config={"reset_noise_scale": 0.01, "kp": 40.0, "kd": 6.0}
        )
        .framework("torch")
        .training(
            # SAC specific parameters
            twin_q=True,
            q_model_config={"fcnet_hiddens": [256, 256]},
            policy_model_config={"fcnet_hiddens": [256, 256]},
            tau=0.005,
            initial_alpha=0.2,
            target_entropy="auto",
            n_step=1,
            replay_buffer_config={
                "capacity": 131072,
                "prioritized_replay_alpha": 0.0,
                "prioritized_replay_beta": 0.4,
            },
            train_batch_size=512,
            target_network_update_freq=1,
            # Learning rates
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
            gamma=0.99,
        )
        .env_runners(
            num_env_runners=12,
            num_envs_per_env_runner=1,
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
            num_cpus_per_learner=4,
        )
        .evaluation(
            evaluation_interval=EVALUATION_INTERVAL,
            evaluation_duration=EVALUATION_EPISODES,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=4,
            evaluation_parallel_to_training=True,
            evaluation_config={
                "explore": False,
                "env_config": {"reset_noise_scale": 0.01, "kp": 40.0, "kd": 6.0}
            },
        )
        .callbacks(MetricsCallback)
        .debugging(seed=42)
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
                checkpoint_frequency=10,
                num_to_keep=3,
                checkpoint_score_attribute="evaluation/env_runners/episode_return_mean",
                checkpoint_score_order="max",
            ),
        ),
    )
    
    results = tuner.fit()
    
    best_result = results.get_best_result(
        metric="evaluation/env_runners/episode_return_mean", 
        mode="max"
    )
    
    if best_result and best_result.checkpoint:
        save_path = P.EXPERT_MODEL_DIR / f"best_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        save_path.mkdir(parents=True, exist_ok=True)
        best_result.checkpoint.to_directory(str(save_path))
        print(f"Best model saved to: {save_path}")
        print(f"Success Rate: {certificate_state['best_success_rate']:.1%}")

if __name__ == "__main__":
    main()