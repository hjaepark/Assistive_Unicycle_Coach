# train/unicycle_train/coach/v0/eval.py


"""
Usage:
   # Baseline (no coach):
   python -m train.unicycle_train.coach.v0.eval --mode no_coach --seed 42

   # With trained coach:
   python -m train.unicycle_train.coach.v0.eval --mode with_coach --coach-path <path_to_coach.pth> --seed 42

   # Run multiple seeds at once:
   python -m train.unicycle_train.coach.v0.eval --mode no_coach --seeds 42 43 44 45 46
   python -m train.unicycle_train.coach.v0.eval --mode with_coach --coach-path <path> --seeds 42 43 44 45 46
"""

import os
import argparse
import time
import json
import numpy as np
import torch
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

TRAIN_TIMESTEPS = 72_000     # Same as human experiment (20 min at 60Hz)
EVAL_TIMESTEPS = 18_000      # Same as human experiment (5 min at 60Hz)

# Student reward weights (match with train_v1.py)
REWARD_WEIGHTS = {
    "posture": 0.0,
    "pendulum": 0.0,
    "speed_reward_weight": 0.0,
    "alive_bonus": 0.0,
    "upright_gate_k": 0.0,
    "progress": 60.0,
    "goal_bonus": 50.0,
    "fall_penalty": -20.0,
}

# Student SAC hyperparameters (match with train.py)
STUDENT_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
}

class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_logs = []
        self.current_episode = 0

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.current_episode += 1
                entry = {
                    "episode": self.current_episode,
                    "timestep": self.num_timesteps,
                    "reward": float(info.get("episode", {}).get("r", 0.0)),
                    "length": int(info.get("episode", {}).get("l", 0)),
                    "goal_reached": bool(info.get("goal_reached", False)),
                    "is_fallen": bool(info.get("is_fallen", False)),
                    "x_position": float(info.get("x_position", 0.0)),
                    "unicycle_roll": float(info.get("unicycle_roll", 0.0)),
                    "unicycle_pitch": float(info.get("unicycle_pitch", 0.0)),
                }
                self.episode_logs.append(entry)
        return True

    def get_logs(self):
        return self.episode_logs

    def reset_logs(self):
        self.episode_logs = []
        self.current_episode = 0

def make_env_no_coach(log_dir, seed):
    base_env = gym.make(
        "UnicyclePendulumPD-v0",
        reward_weights=REWARD_WEIGHTS,
    )
    monitored_env = Monitor(base_env, filename=str(log_dir / "monitor.csv"))
    monitored_env.reset(seed=seed)
    monitored_env.action_space.seed(seed)
    monitored_env.observation_space.seed(seed)
    return monitored_env


def make_env_with_coach(log_dir, seed, coach_agent, coach_buffer):
    base_env = gym.make(
        "UnicyclePendulumPD-v0",
        reward_weights=REWARD_WEIGHTS,
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

def load_trained_coach(coach_path, device="cpu"):
    coach = CoachAgent(
        obs_dim=22,
        action_dim=2,
        lr=0.0,       # Not training, doesn't matter
        device=device,
    )
    coach.load(coach_path)

    # Freeze coach completely
    coach.actor.eval()
    for p in coach.actor.parameters():
        p.requires_grad = False

    return coach

def run_single_seed(mode, seed, coach_path, run_dir, device="cpu"):
    """
    Run one full evaluation (training + eval) for a single seed.

    Args:
        mode: "no_coach" or "with_coach"
        seed: random seed
        coach_path: path to coach checkpoint (only used if mode == "with_coach")
        run_dir: directory to save results
        device: torch device
    
    Returns:
        dict with training and evaluation results
    """
    sb3_set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_dir = seed_dir / "training"
    eval_dir = seed_dir / "evaluation"
    train_dir.mkdir(exist_ok=True)
    eval_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Seed: {seed} | Mode: {mode}")
    print(f"{'='*60}")

    print(f"\n  [Phase 2A] Training ({TRAIN_TIMESTEPS:,} steps)...")
    train_start = time.time()

    if mode == "with_coach":
        coach = load_trained_coach(coach_path, device=device)
        coach_buffer = CoachReplayBuffer(capacity=500_000, obs_dim=22, action_dim=2)
        train_env = make_env_with_coach(train_dir, seed, coach, coach_buffer)
    else:
        train_env = make_env_no_coach(train_dir, seed)

    student = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=STUDENT_CONFIG["learning_rate"],
        buffer_size=STUDENT_CONFIG["buffer_size"],
        learning_starts=STUDENT_CONFIG["learning_starts"],
        batch_size=STUDENT_CONFIG["batch_size"],
        tau=STUDENT_CONFIG["tau"],
        gamma=STUDENT_CONFIG["gamma"],
        device=device,
        verbose=0,
        seed=seed,
    )

    train_logger = EpisodeLogger()
    student.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        callback=train_logger,
        progress_bar=False,
    )

    train_elapsed = time.time() - train_start
    train_logs = train_logger.get_logs()

    # Save trained student
    student_path = seed_dir / "student_after_training"
    student.save(student_path.as_posix())

    train_env.close()

    # Training summary
    train_successes = sum(1 for e in train_logs if e["goal_reached"])
    train_falls = sum(1 for e in train_logs if e["is_fallen"])
    n_train_eps = len(train_logs)

    print(f"    Episodes: {n_train_eps}")
    print(f"    Successes: {train_successes}/{n_train_eps} ({100*train_successes/max(1,n_train_eps):.1f}%)")
    print(f"    Falls: {train_falls}/{n_train_eps} ({100*train_falls/max(1,n_train_eps):.1f}%)")
    print(f"    Wall time: {train_elapsed:.1f}s")

    print(f"\n  [Phase 2B] Evaluation ({EVAL_TIMESTEPS:,} steps, no coach)...")
    eval_start = time.time()

    # Fresh environment without coach
    eval_env = make_env_no_coach(eval_dir, seed + 10000)

    # Load student into new env (important: set_env before predict)
    student.set_env(eval_env)

    eval_logger = EpisodeLogger()
    student.learn(
        total_timesteps=EVAL_TIMESTEPS,
        callback=eval_logger,
        progress_bar=False,
        reset_num_timesteps=False,   # Continue timestep count
    )

    eval_elapsed = time.time() - eval_start
    eval_logs = eval_logger.get_logs()

    eval_env.close()

    # Eval summary
    eval_successes = sum(1 for e in eval_logs if e["goal_reached"])
    eval_falls = sum(1 for e in eval_logs if e["is_fallen"])
    n_eval_eps = len(eval_logs)
    eval_rewards = [e["reward"] for e in eval_logs]
    eval_x_positions = [e["x_position"] for e in eval_logs]

    print(f"    Episodes: {n_eval_eps}")
    print(f"    Successes: {eval_successes}/{n_eval_eps} ({100*eval_successes/max(1,n_eval_eps):.1f}%)")
    print(f"    Falls: {eval_falls}/{n_eval_eps} ({100*eval_falls/max(1,n_eval_eps):.1f}%)")
    print(f"    Mean reward: {np.mean(eval_rewards):.2f}" if eval_rewards else "    No episodes completed")
    print(f"    Mean final x: {np.mean(eval_x_positions):.2f}m" if eval_x_positions else "")
    print(f"    Wall time: {eval_elapsed:.1f}s")

    result = {
        "seed": seed,
        "mode": mode,
        "coach_path": str(coach_path) if coach_path else None,
        "train_timesteps": TRAIN_TIMESTEPS,
        "eval_timesteps": EVAL_TIMESTEPS,

        "training": {
            "episodes": n_train_eps,
            "successes": train_successes,
            "success_rate": train_successes / max(1, n_train_eps),
            "falls": train_falls,
            "fall_rate": train_falls / max(1, n_train_eps),
            "wall_time": train_elapsed,
            "episode_logs": train_logs,
        },

        "evaluation": {
            "episodes": n_eval_eps,
            "successes": eval_successes,
            "success_rate": eval_successes / max(1, n_eval_eps),
            "falls": eval_falls,
            "fall_rate": eval_falls / max(1, n_eval_eps),
            "mean_reward": float(np.mean(eval_rewards)) if eval_rewards else 0.0,
            "mean_x_position": float(np.mean(eval_x_positions)) if eval_x_positions else 0.0,
            "wall_time": eval_elapsed,
            "episode_logs": eval_logs,
        },
    }

    with open(seed_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

def aggregate_results(results, run_dir):
    """Aggregate results across seeds and save summary."""
    mode = results[0]["mode"]
    seeds = [r["seed"] for r in results]

    # Training aggregation
    train_success_rates = [r["training"]["success_rate"] for r in results]
    train_fall_rates = [r["training"]["fall_rate"] for r in results]

    # Evaluation aggregation (this is what matters)
    eval_success_rates = [r["evaluation"]["success_rate"] for r in results]
    eval_fall_rates = [r["evaluation"]["fall_rate"] for r in results]
    eval_mean_rewards = [r["evaluation"]["mean_reward"] for r in results]
    eval_mean_x = [r["evaluation"]["mean_x_position"] for r in results]

    summary = {
        "mode": mode,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "train_timesteps": TRAIN_TIMESTEPS,
        "eval_timesteps": EVAL_TIMESTEPS,

        "training_aggregate": {
            "success_rate_mean": float(np.mean(train_success_rates)),
            "success_rate_std": float(np.std(train_success_rates)),
            "fall_rate_mean": float(np.mean(train_fall_rates)),
            "fall_rate_std": float(np.std(train_fall_rates)),
        },

        "evaluation_aggregate": {
            "success_rate_mean": float(np.mean(eval_success_rates)),
            "success_rate_std": float(np.std(eval_success_rates)),
            "success_rate_per_seed": {s: r for s, r in zip(seeds, eval_success_rates)},
            "fall_rate_mean": float(np.mean(eval_fall_rates)),
            "fall_rate_std": float(np.std(eval_fall_rates)),
            "mean_reward_mean": float(np.mean(eval_mean_rewards)),
            "mean_reward_std": float(np.std(eval_mean_rewards)),
            "mean_x_position_mean": float(np.mean(eval_mean_x)),
            "mean_x_position_std": float(np.std(eval_mean_x)),
        },
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {mode} ({len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"  Training ({TRAIN_TIMESTEPS:,} steps):")
    print(f"    Success rate: {np.mean(train_success_rates)*100:.1f}% +/- {np.std(train_success_rates)*100:.1f}%")
    print(f"    Fall rate:    {np.mean(train_fall_rates)*100:.1f}% +/- {np.std(train_fall_rates)*100:.1f}%")
    print(f"")
    print(f"  Evaluation ({EVAL_TIMESTEPS:,} steps, NO coach):")
    print(f"    Success rate: {np.mean(eval_success_rates)*100:.1f}% +/- {np.std(eval_success_rates)*100:.1f}%")
    print(f"    Fall rate:    {np.mean(eval_fall_rates)*100:.1f}% +/- {np.std(eval_fall_rates)*100:.1f}%")
    print(f"    Mean reward:  {np.mean(eval_mean_rewards):.2f} +/- {np.std(eval_mean_rewards):.2f}")
    print(f"    Mean final x: {np.mean(eval_mean_x):.2f}m +/- {np.std(eval_mean_x):.2f}m")
    print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Coach V0 Evaluation")
    parser.add_argument("--mode", type=str, required=True, choices=["no_coach", "with_coach"],
                        help="Evaluation mode: 'no_coach' (baseline) or 'with_coach' (trained coach)")
    parser.add_argument("--coach-path", type=str, default=None,
                        help="Path to trained coach checkpoint (.pth)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds to run")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for training (cpu or cuda)")
    parser.add_argument("--train-steps", type=int, default=72_000,
                        help="Training timesteps (default: 72000)")
    parser.add_argument("--eval-steps", type=int, default=18_000,
                        help="Evaluation timesteps (default: 18000)")
    args = parser.parse_args()

    global TRAIN_TIMESTEPS, EVAL_TIMESTEPS
    TRAIN_TIMESTEPS = args.train_steps
    EVAL_TIMESTEPS = args.eval_steps

    # Validate arguments
    if args.mode == "with_coach" and args.coach_path is None:
        parser.error("--coach-path is required when --mode is 'with_coach'")

    if args.mode == "with_coach" and not Path(args.coach_path).exists():
        parser.error(f"Coach checkpoint not found: {args.coach_path}")

    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42, 43, 44, 45, 46]

    # Create run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"train/unicycle_train/coach/v0/eval_logs/eval_{args.mode}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save eval config
    eval_config = {
        "mode": args.mode,
        "coach_path": args.coach_path,
        "seeds": seeds,
        "train_timesteps": TRAIN_TIMESTEPS,
        "eval_timesteps": EVAL_TIMESTEPS,
        "reward_weights": REWARD_WEIGHTS,
        "student_config": STUDENT_CONFIG,
        "device": args.device,
    }
    with open(run_dir / "eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)

    print("=" * 60)
    print(f"  Coach V0 Phase 2 Evaluation")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Seeds: {seeds}")
    print(f"  Train: {TRAIN_TIMESTEPS:,} steps")
    print(f"  Eval:  {EVAL_TIMESTEPS:,} steps (no coach)")
    if args.coach_path:
        print(f"  Coach: {args.coach_path}")
    print(f"  Output: {run_dir}")
    print("=" * 60)

    # Run all seeds
    results = []
    for seed in seeds:
        result = run_single_seed(
            mode=args.mode,
            seed=seed,
            coach_path=args.coach_path,
            run_dir=run_dir,
            device=args.device,
        )
        results.append(result)

    # Aggregate and save summary
    summary = aggregate_results(results, run_dir)

    print(f"\n  Results saved to: {run_dir}")


if __name__ == "__main__":
    main()