# train/unicycle_train/student/train.py

from typing import Any, Dict, Callable
from pathlib import Path
import os
import argparse
import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed as sb3_set_seed

from train.unicycle_train.evaluate import EvalCallBack
from train.unicycle_train.cotraining.config_loader import load_config, save_config

import gymnasium as gym
import envs.unicycle
from train.unicycle_train import path as P

# Estimator imports (added for skill tracking)
from train.unicycle_train.coach.estimators.marb_estimator import (
    MARBConfig, MovingAverageRewardBasedEstimator, precompute_params
)

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, 
                   default="train/unicycle_train/cotraining/configs/optimized.yaml",
                   help="Path to config YAML file (reuses cotraining config)")
    p.add_argument("--name", type=str, help="Override experiment name")
    p.add_argument("--seed", type=int, help="Override seed")
    return p


def next_run_dir(base: Path) -> Path:
    """Generate next run directory with timestamp and random suffix"""
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


def make_train_env(log_dir: Path, seed: int):
    """Create training environment with Monitor wrapper"""
    base_env = gym.make("UnicyclePendulumPD-v0", full_obs_in_info=True)

    # Monitor wrapper for episode statistics
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

    monitored_env.reset(seed=seed)
    monitored_env.action_space.seed(seed)
    monitored_env.observation_space.seed(seed)

    return monitored_env


def make_eval_env() -> gym.Env:
    """Create evaluation environment"""
    return gym.make("UnicyclePendulumPD-v0")


class StudentLoggingCallback(BaseCallback):
    """Log student metrics - matches cotraining callback naming"""
    
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        
    def _on_step(self) -> bool:
        # Check if any episode ended
        if len(self.locals.get("dones", [])) > 0:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    
                    # Student episode metrics (matches cotraining)
                    if "episode" in info:
                        self.logger.record("student/rollout/reward", info["episode"]["r"])
                        self.logger.record("student/rollout/length", info["episode"]["l"])
                    
                    # Environment metrics
                    if "goal_reached" in info:
                        self.logger.record("student/rollout/goal_reached", 
                                         float(info["goal_reached"]))
                    if "is_fallen" in info:
                        self.logger.record("student/rollout/is_fallen", 
                                         float(info["is_fallen"]))
                    
                    # Additional metrics for analysis
                    if "unicycle_roll" in info:
                        self.logger.record("student/rollout/unicycle_roll", 
                                         info["unicycle_roll"])
                    if "unicycle_pitch" in info:
                        self.logger.record("student/rollout/unicycle_pitch", 
                                         info["unicycle_pitch"])
                    if "pendulum_ang" in info:
                        self.logger.record("student/rollout/pendulum_ang", 
                                         info["pendulum_ang"])
                    if "x_position" in info:
                        self.logger.record("student/rollout/final_x_position", 
                                         info["x_position"])
                    
                    self.logger.dump(self.num_timesteps)
        
        return True


class SkillEstimationCallback(BaseCallback):
    """Track skill level for baseline (matches frozen coach behavior)"""
    
    def __init__(self, estimator, verbose=0):
        super().__init__(verbose)
        self.estimator = estimator
        self.skill_level = 0.0
        
    def _on_step(self) -> bool:
        # Check if any episode ended
        if len(self.locals.get("dones", [])) > 0:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    
                    # Update skill level using episode return
                    if "episode" in info and "r" in info["episode"]:
                        self.skill_level = self.estimator.update(info["episode"]["r"])
                        
                        # Log skill level (matches frozen coach logging)
                        self.logger.record("coach/rollout/skill_level", self.skill_level)
                        self.logger.dump(self.num_timesteps)
        
        return True


def main():
    args = build_arg_parser().parse_args()

    # Load config (reuses cotraining config!)
    config = load_config(args.config)
    print(f"[info] Loaded config from: {args.config}")
    print(f"[info] Note: Coach/reward sections in config are ignored (baseline mode)")
    
    # Apply overrides
    if args.name:
        config['experiment']['name'] = args.name
    else:
        # Add suffix to distinguish from cotraining
        config['experiment']['name'] = f"{config['experiment']['name']}_baseline"
    
    if args.seed:
        config['training']['seed'] = args.seed
    
    # Generate seed if not specified
    if config['training']['seed'] is None:
        config['training']['seed'] = int.from_bytes(os.urandom(4), "little")
    
    SEED = config['training']['seed']
    sb3_set_seed(SEED)
    
    # Setup directories
    run_dir = next_run_dir(P.LOGS_STUDENT_DIR)
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
    
    # Initialize estimator (uses config['estimator']) - for comparison with frozen coach
    if config['estimator']['type'] == "MARB":
        marb_cfg, info = precompute_params(P.DATA_STUDENT_DIR, k_candidates=range(5, 11))
        print(f"[info] MARB: k={marb_cfg.k}, G_min={info['G_min']:.2f}, G_max={info['G_max']:.2f}")
        estimator = MovingAverageRewardBasedEstimator(marb_cfg)
        print("[info] Skill estimator initialized for logging (baseline has no coach)")
    elif config['estimator']['type'] == "manual":
        class ManualEstimator:
            def __init__(self, skill):
                self.skill = skill
            def update(self, episode_return):
                return self.skill
        estimator = ManualEstimator(config['estimator'].get('manual_skill_level', 0.5))
        print(f"[info] Manual estimator: skill={config['estimator'].get('manual_skill_level', 0.5)}")
    else:
        raise ValueError(f"Unknown estimator: {config['estimator']['type']}")
    
    # Create environment (no coach)
    env = make_train_env(run_dir, SEED)
    print("[info] Training environment created (baseline: no coach)")

    # Initialize student (SB3 SAC) - uses config['student']
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
    
    # Add skill estimation callback for comparison
    skill_estimation_callback = SkillEstimationCallback(estimator=estimator, verbose=0)
    
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
        skill_estimation_callback,  # Added for metric comparison
        evaluation_callback
    ]
    
    # Train
    print(f"[info] Training initiated")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=callbacks,
        progress_bar=True
    )
    
    # Save model
    student_model_path = checkpoint_dir / "student_agent"
    model.save(student_model_path.as_posix())
    print(f"[info] Student model saved to: {student_model_path}")
    
    env.close()
    print("[info] Training completed")


if __name__ == "__main__":
    main()