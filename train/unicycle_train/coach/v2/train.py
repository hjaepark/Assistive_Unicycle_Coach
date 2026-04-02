# train/unicycle_train/coach/v2/train.py

"""
Coach V2 - Constrained Coaching (Action Penalty)

Identical to V1 pipeline with one addition:
Coach reward includes a penalty for total intervention magnitude.

    coach_reward = convergence_reward - lambda_action * mean_coach_action

Changes from V1:
    - Import V2 CoachWrapper (tracks coach action norm)
    - config["reward"]["lambda_action"] controls penalty strength
    - compute_coach_reward() subtracts action penalty

Usage:
    python -m train.unicycle_train.coach.v2.train
    python -m train.unicycle_train.coach.v2.train --bc_checkpoint path/to/best_bc_actor.pth
    python -m train.unicycle_train.coach.v2.train --config train/unicycle_train/coach/v2/configs/default.yaml
"""

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

# [V2 CHANGE] Import V2 wrapper instead of V0
from train.unicycle_train.coach.v2.wrapper import CoachWrapper

DEFAULT_CONFIG = {
    "experiment": {
        "name": "coach_v2",
    },

    "training": {
        "seed": 42,
        "sim_time_max": 1200.0,
        "n_consecutive": 3,
        "coach_gradient_steps": 100,
        "coach_batch_size": 256,
    },

    "coach": {
        "obs_dim": 22,
        "action_dim": 2,
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
        "success_scale": 100.0,
        "failure_penalty": -100.0,
        # [V2 ADDITION] Penalty coefficient for mean coach action norm²
        "lambda_action": 0.5,
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


def find_latest_bc_checkpoint(v1_dir: Path) -> Path:
    """Look for BC checkpoint in V1 logs (reuse V1 BC pretraining)."""
    logs_dir = v1_dir / "logs"
    bc_runs = sorted(logs_dir.glob("bc_*"))
    if not bc_runs:
        raise FileNotFoundError(f"No BC runs found in {logs_dir}")
    checkpoint = bc_runs[-1] / "checkpoints" / "best_bc_actor.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No best_bc_actor.pth in {bc_runs[-1]}")
    return checkpoint


def load_bc_weights(coach: CoachAgent, bc_checkpoint: Path):
    state_dict = torch.load(str(bc_checkpoint), map_location=coach.device)
    coach.actor.load_state_dict(state_dict)
    print(f"[INIT] BC pretrained weights loaded into coach actor")
    print(f"[INIT] Checkpoint: {bc_checkpoint}")
    print(f"[INIT] Note: critic and target networks remain randomly initialized")


def make_student_env(log_dir, seed, coach_agent, coach_buffer):
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
    wrapped_env = CoachWrapper(
        env=monitored_env,
        coach_agent=coach_agent,
        coach_buffer=coach_buffer,
    )
    wrapped_env.reset(seed=seed)
    wrapped_env.action_space.seed(seed)
    wrapped_env.observation_space.seed(seed)
    return wrapped_env


def compute_coach_reward(sim_time, sim_time_max, converged, mean_coach_action, config):
    """
    Coach V2 reward with action penalty.

    convergence_reward:
        Success: (1 - sim_time / sim_time_max) * success_scale
        Failure: failure_penalty

    coach_reward = convergence_reward - lambda_action * mean_coach_action

    Args:
        sim_time: student's convergence time
        sim_time_max: maximum allowed simulation time
        converged: whether the student converged
        mean_coach_action: mean ||coach_action||² over all inner loop steps
        config: full config dict
    """
    # Convergence reward (identical to V0/V1)
    if converged:
        convergence_reward = (1.0 - sim_time / sim_time_max) * config["reward"]["success_scale"]
    else:
        convergence_reward = config["reward"]["failure_penalty"]

    # [V2 ADDITION] Action penalty
    lambda_action = config["reward"]["lambda_action"]
    action_penalty = lambda_action * mean_coach_action

    coach_reward = convergence_reward - action_penalty

    return float(coach_reward), float(convergence_reward), float(action_penalty)


class StopOnConvergence(BaseCallback):
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


def main():
    parser = argparse.ArgumentParser(description="Coach V2 Training (Constrained Coaching)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--bc_checkpoint", type=str, default=None,
                        help="Path to BC pretrained actor (auto-detects from V1 if omitted)")
    parser.add_argument("--max_students", type=int, default=None,
                        help="Stop after N students (default: run until Ctrl+C)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if config["training"]["seed"] is None:
        config["training"]["seed"] = int.from_bytes(os.urandom(4), "little")

    SEED = config["training"]["seed"]
    sb3_set_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Directories
    script_dir = Path(__file__).resolve().parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = script_dir / "logs" / f"v2_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    with open(run_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    (run_dir / "seed.txt").write_text(f"{SEED}\n")

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    # Find BC checkpoint (look in V1 directory)
    if args.bc_checkpoint is not None:
        bc_checkpoint = Path(args.bc_checkpoint)
    else:
        v1_dir = script_dir.parent / "v1"
        bc_checkpoint = find_latest_bc_checkpoint(v1_dir)

    if not bc_checkpoint.exists():
        print(f"[error] BC checkpoint not found: {bc_checkpoint}")
        exit(1)

    print("=" * 70)
    print("[INFO] Coach V2 Training (Constrained Coaching)")
    print("=" * 70)
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Seed: {SEED}")
    print(f"[INFO] BC checkpoint: {bc_checkpoint}")
    print(f"[INFO] Sim time max: {config['training']['sim_time_max']}s")
    print(f"[INFO] Convergence: {config['training']['n_consecutive']} consecutive successes")
    print(f"[INFO] Coach gradient steps per update: {config['training']['coach_gradient_steps']}")
    print(f"[INFO] Lambda action penalty: {config['reward']['lambda_action']}")
    print("=" * 70)

    # Initialize coach (V0 architecture, same as V1)
    coach = CoachAgent(
        obs_dim=config["coach"]["obs_dim"],
        action_dim=config["coach"]["action_dim"],
        lr=config["coach"]["learning_rate"],
        gamma=config["coach"]["gamma"],
        tau=config["coach"]["tau"],
        auto_alpha=config["coach"]["auto_alpha"],
        device=config["coach"]["device"],
    )

    # Load BC pretrained weights into actor
    load_bc_weights(coach, bc_checkpoint)

    coach_buffer = CoachReplayBuffer(
        capacity=config["coach"]["buffer_capacity"],
        obs_dim=config["coach"]["obs_dim"],
        action_dim=config["coach"]["action_dim"],
    )
    print("[INIT] Coach agent and buffer initialized")

    # Save BC checkpoint path for reproducibility
    (run_dir / "bc_checkpoint.txt").write_text(str(bc_checkpoint))

    training_log = []
    student_idx = 0

    try:
        while True:
            student_idx += 1
            # [V2 ADDITION] Stop after max_students if specified
            if args.max_students is not None and student_idx > args.max_students:
                print(f"\n[INFO] Reached max_students={args.max_students}. Stopping.")
                break
            print(f"\n{'='*70}")
            print(f"[INFO] Student #{student_idx}")
            print(f"{'='*70}")

            student_log_dir = run_dir / f"student_{student_idx:04d}"
            student_log_dir.mkdir(exist_ok=True)

            env = make_student_env(
                log_dir=student_log_dir,
                seed=SEED + student_idx,
                coach_agent=coach,
                coach_buffer=coach_buffer,
            )

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

            env.reset_inner_loop()

            inner_start_time = time.time()

            stop_callback = StopOnConvergence(
                env=env,
                n_consecutive=n_consecutive,
                sim_time_max=sim_time_max,
            )

            student.learn(
                total_timesteps=10_000_000,
                callback=stop_callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )

            converged = stop_callback.converged

            inner_elapsed = time.time() - inner_start_time
            sim_time = env.inner_loop_sim_time
            summary = env.get_inner_loop_summary()

            # [V2 CHANGE] Get mean coach action from V2 wrapper summary
            mean_coach_action = summary["inner_loop/mean_coach_action"]

            # [V2 CHANGE] Compute reward with action penalty
            coach_reward, convergence_reward, action_penalty = compute_coach_reward(
                sim_time=sim_time,
                sim_time_max=sim_time_max,
                converged=converged,
                mean_coach_action=mean_coach_action,
                config=config,
            )

            coach_buffer.update_last_reward(coach_reward)

            status = "CONVERGED" if converged else "TIMEOUT"
            print(f"\n  [INFO] Result: {status}")
            print(f"  [INFO] Sim time: {sim_time:.2f}s")
            print(f"  [INFO] Episodes: {env.episode_count}")
            print(f"  [INFO] Convergence reward: {convergence_reward:.2f}")
            print(f"  [INFO] Action penalty: {action_penalty:.4f}")
            print(f"  [INFO] Coach reward (final): {coach_reward:.2f}")
            print(f"  [INFO] Mean coach action: {mean_coach_action:.4f}")
            print(f"  [INFO] Wall time: {inner_elapsed:.1f}s")

            if coach_buffer.size >= config["student"]["learning_starts"]:
                coach_logs = coach.update_from_buffer(
                    coach_buffer,
                    n_gradient_steps=config["training"]["coach_gradient_steps"],
                    batch_size=config["training"]["coach_batch_size"],
                )
                print(f"  [INFO] Coach update: {coach_logs.get('train/critic_loss', 'N/A'):.4f} critic loss")
            else:
                coach_logs = {}
                print(f"  [INFO] Coach update: skipped (buffer size {coach_buffer.size})")

            log_entry = {
                "student_idx": student_idx,
                "converged": converged,
                "sim_time": sim_time,
                "coach_reward": coach_reward,
                "convergence_reward": convergence_reward,
                "action_penalty": action_penalty,
                "mean_coach_action": mean_coach_action,
                "episodes": env.episode_count,
                "steps": env.inner_loop_steps,
                "wall_time": inner_elapsed,
                **summary,
                **coach_logs,
            }
            training_log.append(log_entry)

            # Tensorboard logging
            writer.add_scalar("coach/reward", coach_reward, student_idx)
            writer.add_scalar("coach/convergence_reward", convergence_reward, student_idx)
            writer.add_scalar("coach/action_penalty", action_penalty, student_idx)
            writer.add_scalar("coach/mean_coach_action", mean_coach_action, student_idx)
            writer.add_scalar("coach/sim_time", sim_time, student_idx)
            writer.add_scalar("coach/converged", float(converged), student_idx)
            writer.add_scalar("coach/episodes", env.episode_count, student_idx)
            writer.add_scalar("coach/steps", env.inner_loop_steps, student_idx)
            if coach_logs:
                for k, v in coach_logs.items():
                    writer.add_scalar(k, v, student_idx)
            writer.flush()

            with open(run_dir / "training_log.json", 'w') as f:
                json.dump(training_log, f, indent=2)

            if student_idx % 10 == 0:
                coach.save(models_dir / f"coach_step_{student_idx:04d}.pth")

            env.close()

    except KeyboardInterrupt:
        print("\n\n[INFO] Training stopped by user (Ctrl+C)")

    finally:
        writer.close()

        coach.save(models_dir / "coach_final.pth")
        print(f"  [INFO] Final coach saved: {models_dir / 'coach_final.pth'}")

        with open(run_dir / "training_log.json", 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"[INFO] Training log saved: {run_dir / 'training_log.json'}")

        if training_log:
            rewards = [e["coach_reward"] for e in training_log]
            convergence_rewards = [e["convergence_reward"] for e in training_log]
            action_penalties = [e["action_penalty"] for e in training_log]
            converged_count = sum(1 for e in training_log if e["converged"])
            print(f"\n{'='*70}")
            print(f"  [INFO] Training Summary")
            print(f"{'='*70}")
            print(f"  [INFO] Total students: {len(training_log)}")
            print(f"  [INFO] Converged: {converged_count}/{len(training_log)}")
            print(f"  [INFO] Avg coach reward: {np.mean(rewards):.2f}")
            print(f"  [INFO] Avg convergence reward: {np.mean(convergence_rewards):.2f}")
            print(f"  [INFO] Avg action penalty: {np.mean(action_penalties):.4f}")
            print(f"  [INFO] Last 10 avg reward: {np.mean(rewards[-10:]):.2f}")
            print(f"  [INFO] Lambda action: {config['reward']['lambda_action']}")
            print(f"  [INFO] BC checkpoint used: {bc_checkpoint}")
            print(f"{'='*70}")


if __name__ == "__main__":
    main()