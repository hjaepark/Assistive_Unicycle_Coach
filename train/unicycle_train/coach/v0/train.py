# train/unicycle_train/coach/v0/train.py

import os
import argparse
import time
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed as sb3_set_seed
from stable_baselines3.common.callbacks import BaseCallback

import envs.unicycle

from train.unicycle_train.coach.v0.coach_agent import CoachAgent
from train.unicycle_train.coach.v0.coach_buffer import CoachReplayBuffer
from train.unicycle_train.coach.v0.wrapper import CoachWrapper

DEFAULT_CONFIG = {
    "experiment": {
        "name": "coach_v0",
    },

    "training": {
        "seed": None,                # Auto-generate if None
        "sim_time_max": 1200.0,       # Max simulation time per student (seconds)
        "n_consecutive": 3,          # Consecutive successes to converge
        "coach_gradient_steps": 100,   # Gradient steps per coach update
        "coach_batch_size": 256,
    },

    "coach": {
        "obs_dim": 22,               # state(19) + student_action(2) + r_env(1)
        "action_dim": 2,             # 2D assist force
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "auto_alpha": True,
        "buffer_capacity": 1_000_000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },

    "student": {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
    },

    "reward": {
        "success_scale": 100.0,      # r^c = (1 - sim_time/sim_time_max) * scale
        "failure_penalty": -100.0,   # r^c when student fails to converge
    },
}

def load_config(config_path=None):
    config = DEFAULT_CONFIG.copy()

    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        for section, values in user_config.items():
            if section in config and isinstance(values, dict):
                config[section].update(values)
            else:
                config[section] = values

    return config

def make_student_env(log_dir, seed, coach_agent, coach_buffer):
    """
    Create a new environment for a new student.
    """

    # Match student reward weights from train_v1.py
    max_success_rate_weights = {
        "posture": 0.0,
        "pendulum": 0.0,
        "speed_reward_weight": 0.0,
        "alive_bonus": 0.0,
        "upright_gate_k": 0.0,
        "progress": 60.0,
        "goal_bonus": 50.0,
        "fall_penalty": -20.0,
    }

    base_env = gym.make(
        "UnicyclePendulumPD-v0",
        reward_weights=max_success_rate_weights,
    )

    monitored_env = Monitor(base_env, filename=str(log_dir / "monitor.csv"))

    # Coach wrapper
    wrapped_env = CoachWrapper(
        env=monitored_env,
        coach_agent=coach_agent,
        coach_buffer=coach_buffer,
    )

    wrapped_env.reset(seed=seed)
    wrapped_env.action_space.seed(seed)
    wrapped_env.observation_space.seed(seed)

    return wrapped_env

def compute_coach_reward(sim_time, sim_time_max, converged, config):
    """
    Compute coach reward based on student's convergence time.
    
    Success: r^c = (1 - sim_time / sim_time_max) * scale
    Failure: r^c = failure_penalty
    """
    if converged:
        reward = (1.0 - sim_time / sim_time_max) * config["reward"]["success_scale"]
    else:
        reward = config["reward"]["failure_penalty"]

    return float(reward)

class StopOnConvergence(BaseCallback):
    """Stop student training on 3 consecutive successes or sim_time_max."""
    
    def __init__(self, env, n_consecutive=3, sim_time_max=1200.0):
        super().__init__()
        self.env = env
        self.n_consecutive = n_consecutive
        self.sim_time_max = sim_time_max
        self.converged = False
    
    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                if self.env.is_converged(self.n_consecutive):
                    self.converged = True
                    return False
                
                if self.env.is_timed_out(self.sim_time_max):
                    return False
        
        return True

# Main outer loop
def main():
    parser = argparse.ArgumentParser(description="Coach V0 Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Seed
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if config["training"]["seed"] is None:
        config["training"]["seed"] = int.from_bytes(os.urandom(4), "little")

    SEED = config["training"]["seed"]
    sb3_set_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Directories
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"train/unicycle_train/coach/v0/logs/run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Save config
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    (run_dir / "seed.txt").write_text(f"{SEED}\n")

    # Tensorboard
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    print("=" * 70)
    print("[info] Coach V0 Training")
    print("=" * 70)
    print(f"[info] Run dir: {run_dir}")
    print(f"[info] Seed: {SEED}")
    print(f"[info] Sim time max: {config['training']['sim_time_max']}s")
    print(f"[info] Convergence: {config['training']['n_consecutive']} consecutive successes")
    print(f"[info] Coach gradient steps per update: {config['training']['coach_gradient_steps']}")
    print("=" * 70)

    coach = CoachAgent(
        obs_dim=config["coach"]["obs_dim"],
        action_dim=config["coach"]["action_dim"],
        lr=config["coach"]["learning_rate"],
        gamma=config["coach"]["gamma"],
        tau=config["coach"]["tau"],
        auto_alpha=config["coach"]["auto_alpha"],
        device=config["coach"]["device"],
    )
    coach_buffer = CoachReplayBuffer(
        capacity=config["coach"]["buffer_capacity"],
        obs_dim=config["coach"]["obs_dim"],
        action_dim=config["coach"]["action_dim"],
    )
    print("[init] Coach agent and buffer initialized")

    training_log = []

    student_idx = 0

    try:
        while True:  # Manual termination (Ctrl+C)
            student_idx += 1
            print(f"\n{'='*70}")
            print(f"[info] Student #{student_idx}")
            print(f"{'='*70}")

            # Create new student environment with coach wrapper
            student_log_dir = run_dir / f"student_{student_idx:04d}"
            student_log_dir.mkdir(exist_ok=True)

            env = make_student_env(
                log_dir=student_log_dir,
                seed=SEED + student_idx,
                coach_agent=coach,
                coach_buffer=coach_buffer,
            )

            # Create new student (SB3 SAC)
            student = SAC(
                "MlpPolicy",
                env,
                learning_rate=config["student"]["learning_rate"],
                buffer_size=config["student"]["buffer_size"],
                learning_starts=config["student"]["learning_starts"],
                batch_size=config["student"]["batch_size"],
                tau=config["student"]["tau"],
                gamma=config["student"]["gamma"],
                device="cpu",
                verbose=0,
                seed=SEED + student_idx,
            )
            
            sim_time_max = config["training"]["sim_time_max"]
            n_consecutive = config["training"]["n_consecutive"]
            
            # Reset inner loop
            env.reset_inner_loop()
            
            inner_start_time = time.time()
            
            # Callback stops learn() on convergence or timeout
            stop_callback = StopOnConvergence(
                env=env,
                n_consecutive=n_consecutive,
                sim_time_max=sim_time_max,
            )
            
            # Student learns until callback stops it
            student.learn(
                total_timesteps=10_000_000,  # Arbitrary large number, callback will stop early
                callback=stop_callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            
            converged = stop_callback.converged
            
            inner_elapsed = time.time() - inner_start_time
            sim_time = env.inner_loop_sim_time
            summary = env.get_inner_loop_summary()

            # Compute Coach Reward
            coach_reward = compute_coach_reward(
                sim_time=sim_time,
                sim_time_max=sim_time_max,
                converged=converged,
                config=config,
            )

            # Update last transition's reward in buffer
            coach_buffer.update_last_reward(coach_reward)

            status = "CONVERGED" if converged else "TIMEOUT"
            print(f"\n  [info] Result: {status}")
            print(f"  [info] Sim time: {sim_time:.2f}s")
            print(f"  [info] Episodes: {env.episode_count}")
            print(f"  [info] Coach reward: {coach_reward:.2f}")
            print(f"  [info] Wall time: {inner_elapsed:.1f}s")

            # Coach Update
            if coach_buffer.size >= config["student"]["learning_starts"]:
                coach_logs = coach.update_from_buffer(
                    coach_buffer,
                    n_gradient_steps=config["training"]["coach_gradient_steps"],
                    batch_size=config["training"]["coach_batch_size"],
                )
                print(f"  [info] Coach update: {coach_logs.get('train/critic_loss', 'N/A'):.4f} critic loss")
            else:
                coach_logs = {}
                print(f"  [info] Coach update: skipped (buffer size {coach_buffer.size})")

            log_entry = {
                "student_idx": student_idx,
                "converged": converged,
                "sim_time": sim_time,
                "coach_reward": coach_reward,
                "episodes": env.episode_count,
                "steps": env.inner_loop_steps,
                "wall_time": inner_elapsed,
                **summary,
                **coach_logs,
            }
            training_log.append(log_entry)

            # Tensorboard logging
            writer.add_scalar("coach/reward", coach_reward, student_idx)
            writer.add_scalar("coach/sim_time", sim_time, student_idx)
            writer.add_scalar("coach/converged", float(converged), student_idx)
            writer.add_scalar("coach/episodes", env.episode_count, student_idx)
            writer.add_scalar("coach/steps", env.inner_loop_steps, student_idx)
            if coach_logs:
                for k, v in coach_logs.items():
                    writer.add_scalar(k, v, student_idx)
            writer.flush()

            # Save log periodically
            with open(run_dir / "training_log.json", 'w') as f:
                json.dump(training_log, f, indent=2)

            # Save coach model periodically
            if student_idx % 10 == 0:
                coach.save(models_dir / f"coach_step_{student_idx:04d}.pth")

            # Cleanup
            env.close()

    except KeyboardInterrupt:
        print("\n\n[info] Training stopped by user (Ctrl+C)")

    finally:
        writer.close()

        # Save final coach model
        coach.save(models_dir / "coach_final.pth")
        print(f"  [info] Final coach saved: {models_dir / 'coach_final.pth'}")

        # Save final log
        with open(run_dir / "training_log.json", 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"[info] Training log saved: {run_dir / 'training_log.json'}")

        # Print summary
        if training_log:
            rewards = [e["coach_reward"] for e in training_log]
            converged_count = sum(1 for e in training_log if e["converged"])
            print(f"\n{'='*70}")
            print(f"  [info] Training Summary")
            print(f"{'='*70}")
            print(f"  [info] Total students: {len(training_log)}")
            print(f"  [info] Converged: {converged_count}/{len(training_log)}")
            print(f"  [info] Avg coach reward: {np.mean(rewards):.2f}")
            print(f"  [info] Last 10 avg: {np.mean(rewards[-10:]):.2f}")
            print(f"{'='*70}")


if __name__ == "__main__":
    main()