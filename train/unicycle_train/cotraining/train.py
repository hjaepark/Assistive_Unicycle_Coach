# train/unicycle_train/cotraining/train.py

from typing import Any, Dict, Callable
from pathlib import Path
import os
import argparse
import torch
import numpy as np
import json

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed as sb3_set_seed

from train.unicycle_train.evaluate import EvalCallBack

import gymnasium as gym
import envs.unicycle
from train.unicycle_train import path as P

# Coach imports
from train.unicycle_train.coach.coach_wrapper import CoachWrapper
from train.unicycle_train.coach.coach_agent import CoachAgent
from train.unicycle_train.coach.estimators.marb_estimator import (
    MARBConfig, MovingAverageRewardBasedEstimator, precompute_params
)
from train.unicycle_train.coach.coach_config import CoachRewardConfig

from train.unicycle_train.cotraining.config_loader import load_config, save_config

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, 
                   default="train/unicycle_train/cotraining/configs/default.yaml",
                   help="Path to config YAML file")
    p.add_argument("--name", type=str, help="Override experiment name")
    p.add_argument("--seed", type=int, help="Override seed")
    return p


def next_run_dir(base: Path) -> Path:
    import time
    import random
    
    base.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    random_suffix = random.randint(1000, 9999)
    rd = base / f"run_{timestamp}_{random_suffix}"
    
    max_retries = 10
    for attempt in range(max_retries):
        try:
            rd.mkdir(parents=True, exist_ok=False)
            return rd
        except FileExistsError:
            if attempt < max_retries - 1:
                random_suffix = random.randint(1000, 9999)
                rd = base / f"run_{timestamp}_{random_suffix}"
            else:
                raise
    
    return rd


def make_train_env_with_coach(log_dir: Path, seed: int, coach_agent, estimator, reward_config: CoachRewardConfig = None):
    # Create training environment with coach wrapper
    base_env = gym.make("UnicyclePendulumPD-v0", full_obs_in_info=True)
    
    # 1. Monitor wrapper (adds info["episode"])
    obs_dim = int(np.prod(base_env.observation_space.shape))
    obs_keys = tuple(f"obs_{i}" for i in range(obs_dim))
    extra_keys = (
        "timestamp", "num_steps", "num_episodes",
        "unicycle_roll", "unicycle_pitch", "unicycle_yaw", "pendulum_ang",
        "num_success", "num_falls", "reward", "goal_reached", "is_fallen",
    )
    
    monitored_env = Monitor(
        base_env,
        filename=str(log_dir / "monitor.csv"),
        info_keywords=obs_keys + extra_keys,
    )
    
    # 2. Coach wrapper (on top of Monitor)
    env = CoachWrapper(
        monitored_env,
        coach_agent=coach_agent,
        estimator=estimator,
        reward_config=reward_config
    )
    
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    return env


def make_eval_env() -> gym.Env:
    # Evaluation environment without coach
    return gym.make("UnicyclePendulumPD-v0")


class StudentLoggingCallback(BaseCallback):
    """Log student metrics and track coach rewards"""
    
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.coach_rewards = []  # Track coach episode rewards
        
    def _on_step(self) -> bool:
        if len(self.locals.get("dones", [])) > 0:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    
                    # Student episode metrics
                    if "episode" in info:
                        self.logger.record("student/rollout/reward", info["episode"]["r"])
                        self.logger.record("student/rollout/length", info["episode"]["l"])
                    
                    # Coach rollout metrics (from CoachWrapper)
                    if "rollout/coach_reward_mean" in info:
                        # Store for final statistics
                        self.coach_rewards.append(info["rollout/coach_reward_sum"])
                        
                        # Log to tensorboard
                        self.logger.record("coach/rollout/reward_mean", 
                                         info["rollout/coach_reward_mean"])
                        self.logger.record("coach/rollout/reward_sum", 
                                         info["rollout/coach_reward_sum"])
                        self.logger.record("coach/rollout/weight_assist_mean", 
                                         info["rollout/weight_assist_mean"])
                        self.logger.record("coach/rollout/weight_assist_std", 
                                         info["rollout/weight_assist_std"])
                        self.logger.record("coach/rollout/effective_intervention_norm_mean",
                                         info["rollout/effective_intervention_norm_mean"])
                        self.logger.record("coach/rollout/skill_level", 
                                         info["rollout/skill_level"])
                    
                    # Environment metrics
                    if "goal_reached" in info:
                        self.logger.record("student/rollout/goal_reached", 
                                         float(info["goal_reached"]))
                    if "is_fallen" in info:
                        self.logger.record("student/rollout/is_fallen", 
                                         float(info["is_fallen"]))
                    
                    self.logger.dump(self.num_timesteps)
        
        return True
    
    def _on_training_end(self) -> None:
        """Save coach reward statistics at training end"""
        if self.coach_rewards:
            stats = {
                "mean_coach_reward": float(np.mean(self.coach_rewards)),
                "std_coach_reward": float(np.std(self.coach_rewards)),
                "total_episodes": len(self.coach_rewards),
                "last_100_mean": float(np.mean(self.coach_rewards[-100:])) 
                    if len(self.coach_rewards) >= 100 
                    else float(np.mean(self.coach_rewards)),
                "last_50_mean": float(np.mean(self.coach_rewards[-50:])) 
                    if len(self.coach_rewards) >= 50 
                    else float(np.mean(self.coach_rewards)),
            }
            
            stats_path = Path(self.log_dir) / "coach_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"[info] Coach statistics saved: {stats_path}")
            print(f"[info] Mean coach reward: {stats['mean_coach_reward']:.4f}")
            print(f"[info] Last 100 episodes mean: {stats['last_100_mean']:.4f}")

class CoachTrainingCallback(BaseCallback):
    # Periodically train the coach
    
    def __init__(self, 
                 coach_wrapper,
                 update_freq=1,
                 gradient_steps=10,
                 learning_starts=1000,
                 verbose=0):
        super().__init__(verbose)
        self.coach_wrapper = coach_wrapper
        self.update_freq = update_freq
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if len(self.locals.get("dones", [])) > 0:
            for done in self.locals["dones"]:
                if done:
                    self.episode_count += 1
                    
                    # Update coach every N episodes
                    if self.episode_count % self.update_freq == 0:
                        coach_logs = self.coach_wrapper.update_coach(
                            n_gradient_steps=self.gradient_steps,
                            learning_starts=self.learning_starts
                        )
                        
                        # Log coach training metrics
                        if coach_logs:
                            for key, value in coach_logs.items():
                                self.logger.record(f"coach/{key}", value)
                            
                            self.logger.dump(self.num_timesteps)
        
        return True


def main():
    args = build_arg_parser().parse_args()

    # Load config
    config = load_config(args.config)
    print(f"[info] Loaded config from: {args.config}")
    
    # Apply overrides
    if args.name:
        config['experiment']['name'] = args.name
    if args.seed:
        config['training']['seed'] = args.seed
    
    # Generate seed if not specified
    if config['training']['seed'] is None:
        config['training']['seed'] = int.from_bytes(os.urandom(4), "little")
    
    SEED = config['training']['seed']
    sb3_set_seed(SEED)
    
    # Setup directories
    run_dir = next_run_dir(P.LOGS_COTRAINING_DIR)  # New directory for cotraining
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir = run_dir / "models"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, str(run_dir / "config.yaml"))
    (run_dir / "seed.txt").write_text(f"{SEED}\n")
    
    print(f"[info] Experiment: {config['experiment']['name']}")
    print(f"[info] Run dir: {run_dir}")
    print(f"[info] Seed: {SEED}")
    print(f"[info] Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"[info] Early stopping: {config['training']['allow_early_stopping']}")
    
    # Initialize coach agent
    coach = CoachAgent(
        obs_dim=config['coach']['obs_dim'],
        action_dim=config['coach']['action_dim'],
        lr=config['coach']['learning_rate'],
        gamma=config['coach']['gamma'],
        tau=config['coach']['tau'],
        auto_alpha=config['coach']['auto_alpha'],
        device=config['coach']['device']
    )
    print("[info] Coach agent initialized")
    
    # Initialize estimator
    if config['estimator']['type'] == "MARB":
        marb_cfg, info = precompute_params(P.DATA_STUDENT_DIR, k_candidates=range(5, 11))
        print(f"[info] MARB: k={marb_cfg.k}, G_min={info['G_min']:.2f}, G_max={info['G_max']:.2f}")
        estimator = MovingAverageRewardBasedEstimator(marb_cfg)
    elif config['estimator']['type'] == "manual":
        class ManualEstimator:
            def __init__(self, skill):
                self.skill = skill
            def update(self, episode_return):
                return self.skill
        estimator = ManualEstimator(config['estimator']['manual_skill_level'])
    else:
        raise ValueError(f"Unknown estimator: {config['estimator']['type']}")
    
    # Initialize reward config from YAML
    reward_config = CoachRewardConfig(
        rho=config['reward']['rho'],
        beta=config['reward']['beta'],
        lambda_E=config['reward']['lambda_E'],
        kappa=np.array(config['reward']['kappa'], dtype=np.float32)
    )

    # Create training environment with coach
    env = make_train_env_with_coach(run_dir, SEED, coach, estimator, reward_config)
    print("[info] Training environment with coach created")
    
    # Initialize student (SB3 SAC)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=config['student']['learning_rate'],
        buffer_size=config['student']['buffer_size'],
        learning_starts=config['student']['learning_starts'],
        batch_size=config['student']['batch_size'],
        tau=config['student']['tau'],
        gamma=config['student']['gamma'],
        device="cpu",
        verbose=1,
        seed=SEED,
        tensorboard_log=str(tensorboard_dir),
    )
    model.set_random_seed(SEED)
    print("[info] Student agent (SB3 SAC) initialized")
    
    # Setup callbacks
    student_logging_callback = StudentLoggingCallback(log_dir=run_dir, verbose=0)
    
    coach_training_callback = CoachTrainingCallback(
        coach_wrapper=env,
        update_freq=config['coach']['update_freq'],
        gradient_steps=config['coach']['gradient_steps'],
        learning_starts=config['coach']['learning_starts'],
        verbose=1
    )
    
    evaluation_callback = EvalCallBack(
        make_env_fn=make_eval_env,
        eval_interval_steps=config['training']['eval_interval'],
        n_eval_episodes=config['training']['eval_episodes'],
        target_success_rate=config['training']['target_success_rate'],
        best_model_path=str(checkpoint_dir / "best_student_model"),
        deterministic=True,
        seed=None,
        verbose=1,
        require_consecutive=config['training']['require_consecutive'],
        allow_early_stopping=config['training']['allow_early_stopping'],
    )
    
    callbacks = [
        student_logging_callback,
        coach_training_callback,
        evaluation_callback
    ]
    
    # Train
    print(f"[info] Training initiated")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=callbacks,
        progress_bar=True
    )
    
    # Save models
    student_model_path = checkpoint_dir / "student_agent"
    model.save(student_model_path.as_posix())
    print(f"[info] Student model saved to: {student_model_path}")
    
    # Save coach model
    coach_model_path = checkpoint_dir / "coach_agent.pth"
    torch.save({
        'actor': coach.actor.state_dict(),
        'critic1': coach.critic1.state_dict(),
        'critic2': coach.critic2.state_dict(),
    }, coach_model_path)
    print(f"[info] Coach model saved to: {coach_model_path}")
    
    env.close()
    print("[info] Training completed")


if __name__ == "__main__":
    main()