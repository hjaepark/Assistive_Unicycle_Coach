# train/unicycle_train/coach/v4/eval.py

"""
Coach V4 Evaluation

Evaluates trained V4 coach (π_I from multi-student training + π_A BC frozen) by:
  Phase A: Train a NEW student with frozen coach for TRAIN_EPISODES
           using coaching → self-practice cycle (same as V3 eval)
  Phase B: Remove coach, evaluate student alone for EVAL_EPISODES

Runs both "with_coach" and "no_coach" modes for direct comparison.
Multiple seeds supported for held-out testing.

Uses V4's own intervention_policy, wrapper, coach_agent (independent from V3).

Usage:
    # Both modes (with coach + no coach baseline):
    python -m train.unicycle_train.coach.v4.eval \
        --pi-path train/unicycle_train/coach/v4/logs/v4_20260421_195950_pool100/models/pi_I_iter_0100.pth \
        --seeds 500 600 700 800 900

    # With coach only:
    python -m train.unicycle_train.coach.v4.eval --mode with_coach --pi-path <path> --seed 500

    # No coach baseline only:
    python -m train.unicycle_train.coach.v4.eval --mode no_coach --seed 500
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

# V4 components (independent from V3)
from train.unicycle_train.coach.v4.coach_agent import CoachAgent
from train.unicycle_train.coach.v4.intervention_policy import InterventionPolicy
from train.unicycle_train.coach.v4.wrapper import CoachWrapper


TRAIN_EPISODES = 500
EVAL_EPISODES = 100
K = 10
M = 10

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

STUDENT_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 1_000,
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
                }
                self.episode_logs.append(entry)
        return True


class StopAfterEpisodes(BaseCallback):
    def __init__(self, max_episodes: int):
        super().__init__()
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self):
        for done in self.locals["dones"]:
            if done:
                self.episode_count += 1
                if self.episode_count >= self.max_episodes:
                    return False
        return True


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


def make_env_with_v4_coach(log_dir, seed, assistance_policy):
    base_env = gym.make(
        "UnicyclePendulumPD-v0",
        reward_weights=REWARD_WEIGHTS,
    )
    monitored_env = Monitor(base_env, filename=str(log_dir / "monitor.csv"))

    wrapped_env = CoachWrapper(
        env=monitored_env,
        assistance_policy=assistance_policy,
        w_t=0,  # Set by π_I each iteration
    )
    wrapped_env.reset(seed=seed)
    wrapped_env.action_space.seed(seed)
    wrapped_env.observation_space.seed(seed)
    return wrapped_env


def load_assistance_policy(bc_checkpoint, device="cpu"):
    coach = CoachAgent(
        obs_dim=22,
        action_dim=2,
        lr=0.0,
        device=device,
    )
    state_dict = torch.load(str(bc_checkpoint), map_location=device)
    coach.actor.load_state_dict(state_dict)
    coach.actor.eval()
    for p in coach.actor.parameters():
        p.requires_grad = False
    return coach


def load_intervention_policy(pi_path, device="cpu"):
    pi_I = InterventionPolicy(
        obs_dim=2,
        lr=0.0,
        device=device,
    )
    pi_I.load(pi_path)
    pi_I.actor.eval()
    return pi_I


def find_latest_bc_checkpoint(v1_dir: Path) -> Path:
    logs_dir = v1_dir / "logs"
    bc_runs = sorted(logs_dir.glob("bc_*"))
    if not bc_runs:
        raise FileNotFoundError(f"No BC runs found in {logs_dir}")
    checkpoint = bc_runs[-1] / "checkpoints" / "best_bc_actor.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No best_bc_actor.pth in {bc_runs[-1]}")
    return checkpoint


def train_with_v4_coach(student, env, intervention_policy, train_episodes,
                        coaching_K=K, practice_M=M):
    """
    Phase A: train student with frozen V4 coach (coaching → self-practice cycle).
    Identical to V3 eval logic; V4 π_I is used the same way at inference time.
    """
    all_logs = []
    episodes_done = 0
    pi_obs = np.array([0.0, 0.0], dtype=np.float32)

    while episodes_done < train_episodes:
        # π_I decides w_t (deterministic for eval)
        w_t, _, _ = intervention_policy.predict(pi_obs, deterministic=True)
        env.set_intervention(w_t)

        env.set_mode("coaching")
        remaining = train_episodes - episodes_done
        chunk_K = min(coaching_K, remaining)

        stop_cb = StopAfterEpisodes(max_episodes=chunk_K)
        logger = EpisodeLogger()

        obs, _ = env.reset()
        if student._last_obs is not None:
            student._last_obs = np.array([obs])

        student.learn(
            total_timesteps=10_000_000,
            callback=[stop_cb, logger],
            reset_num_timesteps=False,
            progress_bar=False,
        )

        for entry in logger.episode_logs:
            entry["w_t"] = w_t
            entry["phase"] = "coaching"
            all_logs.append(entry)

        episodes_done += stop_cb.episode_count

        # Update π_I observation from coaching phase summary
        pi_obs = env.get_pi_observation()

        if episodes_done >= train_episodes:
            break

        # Self-practice phase (student continues learning, coach off)
        env.set_mode("eval")
        remaining = train_episodes - episodes_done
        chunk_M = min(practice_M, remaining)

        stop_cb_sp = StopAfterEpisodes(max_episodes=chunk_M)
        logger_sp = EpisodeLogger()

        obs, _ = env.reset()
        if student._last_obs is not None:
            student._last_obs = np.array([obs])

        student.learn(
            total_timesteps=10_000_000,
            callback=[stop_cb_sp, logger_sp],
            reset_num_timesteps=False,
            progress_bar=False,
        )

        for entry in logger_sp.episode_logs:
            entry["w_t"] = -1  # No coach
            entry["phase"] = "self_practice"
            all_logs.append(entry)

        episodes_done += stop_cb_sp.episode_count

        # Update π_I observation from self-practice
        pi_obs = env.get_pi_observation()

    return all_logs


def evaluate_independent(student, eval_episodes, seed, eval_dir, device="cpu"):
    """Phase B: student alone, no coach."""
    eval_env = make_env_no_coach(eval_dir, seed)
    student.set_env(eval_env)

    logger = EpisodeLogger()
    stop_cb = StopAfterEpisodes(max_episodes=eval_episodes)

    student.learn(
        total_timesteps=10_000_000,
        callback=[stop_cb, logger],
        reset_num_timesteps=False,
        progress_bar=False,
    )

    eval_env.close()
    return logger.episode_logs


def run_single_seed(mode, seed, pi_path, bc_checkpoint, run_dir,
                    train_episodes, eval_episodes, device="cpu"):
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

    print(f"\n  [Phase A] Training ({train_episodes} episodes)...")
    train_start = time.time()

    if mode == "with_coach":
        assistance_policy = load_assistance_policy(bc_checkpoint, device=device)
        pi_I = load_intervention_policy(pi_path, device=device)

        train_env = make_env_with_v4_coach(train_dir, seed, assistance_policy)

        student = SAC(
            "MlpPolicy",
            train_env,
            **STUDENT_CONFIG,
            device=device,
            verbose=0,
            seed=seed,
        )

        train_logs = train_with_v4_coach(
            student, train_env, pi_I, train_episodes
        )
        train_env.close()

    else:
        # No coach baseline
        train_env = make_env_no_coach(train_dir, seed)

        student = SAC(
            "MlpPolicy",
            train_env,
            **STUDENT_CONFIG,
            device=device,
            verbose=0,
            seed=seed,
        )

        logger = EpisodeLogger()
        stop_cb = StopAfterEpisodes(max_episodes=train_episodes)
        student.learn(
            total_timesteps=10_000_000,
            callback=[stop_cb, logger],
            progress_bar=False,
        )
        train_logs = logger.episode_logs
        train_env.close()

    train_elapsed = time.time() - train_start

    train_successes = sum(1 for e in train_logs if e["goal_reached"])
    train_falls = sum(1 for e in train_logs if e["is_fallen"])
    n_train_eps = len(train_logs)

    print(f"    Episodes: {n_train_eps}")
    print(f"    Successes: {train_successes}/{n_train_eps} "
          f"({100*train_successes/max(1,n_train_eps):.1f}%)")
    print(f"    Falls: {train_falls}/{n_train_eps} "
          f"({100*train_falls/max(1,n_train_eps):.1f}%)")
    print(f"    Wall time: {train_elapsed:.1f}s")

    student_path = seed_dir / "student_after_training"
    student.save(student_path.as_posix())

    print(f"\n  [Phase B] Evaluation ({eval_episodes} episodes, NO coach)...")
    eval_start = time.time()

    eval_logs = evaluate_independent(
        student, eval_episodes, seed + 10000, eval_dir, device=device
    )

    eval_elapsed = time.time() - eval_start

    eval_successes = sum(1 for e in eval_logs if e["goal_reached"])
    eval_falls = sum(1 for e in eval_logs if e["is_fallen"])
    n_eval_eps = len(eval_logs)
    eval_rewards = [e["reward"] for e in eval_logs]
    eval_x_positions = [e["x_position"] for e in eval_logs]

    print(f"    Episodes: {n_eval_eps}")
    print(f"    Successes: {eval_successes}/{n_eval_eps} "
          f"({100*eval_successes/max(1,n_eval_eps):.1f}%)")
    print(f"    Falls: {eval_falls}/{n_eval_eps} "
          f"({100*eval_falls/max(1,n_eval_eps):.1f}%)")
    if eval_rewards:
        print(f"    Mean reward: {np.mean(eval_rewards):.2f}")
        print(f"    Mean final x: {np.mean(eval_x_positions):.2f}m")
    print(f"    Wall time: {eval_elapsed:.1f}s")

    result = {
        "seed": seed,
        "mode": mode,
        "pi_path": str(pi_path) if pi_path else None,
        "bc_checkpoint": str(bc_checkpoint) if bc_checkpoint else None,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,

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


def aggregate_results(results, run_dir, mode):
    seeds = [r["seed"] for r in results]

    train_success_rates = [r["training"]["success_rate"] for r in results]
    train_fall_rates = [r["training"]["fall_rate"] for r in results]

    eval_success_rates = [r["evaluation"]["success_rate"] for r in results]
    eval_fall_rates = [r["evaluation"]["fall_rate"] for r in results]
    eval_mean_rewards = [r["evaluation"]["mean_reward"] for r in results]
    eval_mean_x = [r["evaluation"]["mean_x_position"] for r in results]

    summary = {
        "mode": mode,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "train_episodes": results[0]["train_episodes"],
        "eval_episodes": results[0]["eval_episodes"],

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
            "mean_x_position_std": float(np.mean(eval_mean_x)),
        },
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {mode} ({len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"  Training ({results[0]['train_episodes']} episodes):")
    print(f"    Success rate: {np.mean(train_success_rates)*100:.1f}% "
          f"+/- {np.std(train_success_rates)*100:.1f}%")
    print(f"    Fall rate:    {np.mean(train_fall_rates)*100:.1f}% "
          f"+/- {np.std(train_fall_rates)*100:.1f}%")
    print()
    print(f"  Evaluation ({results[0]['eval_episodes']} episodes, NO coach):")
    print(f"    Success rate: {np.mean(eval_success_rates)*100:.1f}% "
          f"+/- {np.std(eval_success_rates)*100:.1f}%")
    print(f"    Fall rate:    {np.mean(eval_fall_rates)*100:.1f}% "
          f"+/- {np.std(eval_fall_rates)*100:.1f}%")
    print(f"    Mean reward:  {np.mean(eval_mean_rewards):.2f} "
          f"+/- {np.std(eval_mean_rewards):.2f}")
    print(f"    Mean final x: {np.mean(eval_mean_x):.2f}m "
          f"+/- {np.std(eval_mean_x):.2f}m")
    print(f"    Per-seed success:")
    for s, sr in zip(seeds, eval_success_rates):
        print(f"      seed {s}: {sr*100:.1f}%")
    print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Coach V4 Evaluation")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["no_coach", "with_coach", "both"],
                        help="Evaluation mode (default: both)")
    parser.add_argument("--pi-path", type=str, default=None,
                        help="Path to trained V4 π_I checkpoint (.pth)")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Path to BC pretrained actor (auto-detects from V1 if omitted)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (default: cpu)")
    parser.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag for output dir naming")
    args = parser.parse_args()

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [500, 600, 700, 800, 900]

    script_dir = Path(__file__).resolve().parent

    # BC checkpoint (π_A)
    if args.bc_checkpoint is not None:
        bc_checkpoint = Path(args.bc_checkpoint)
    else:
        v1_dir = script_dir.parent / "v1"
        bc_checkpoint = find_latest_bc_checkpoint(v1_dir)

    # π_I checkpoint
    pi_path = None
    if args.mode in ("with_coach", "both"):
        if args.pi_path is None:
            parser.error("--pi-path required for with_coach/both modes")
        pi_path = Path(args.pi_path)
        if not pi_path.exists():
            parser.error(f"π_I checkpoint not found: {pi_path}")

    modes = ["no_coach", "with_coach"] if args.mode == "both" else [args.mode]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag_str = f"_{args.tag}" if args.tag else ""
    run_dir = script_dir / "eval_logs" / f"eval_{timestamp}{tag_str}"
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_config = {
        "modes": modes,
        "pi_path": str(pi_path) if pi_path else None,
        "bc_checkpoint": str(bc_checkpoint),
        "seeds": seeds,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "coaching_K": K,
        "practice_M": M,
        "reward_weights": REWARD_WEIGHTS,
        "student_config": STUDENT_CONFIG,
        "device": args.device,
    }
    with open(run_dir / "eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)

    print("=" * 60)
    print("  Coach V4 Evaluation")
    print("=" * 60)
    print(f"  Modes:  {modes}")
    print(f"  Seeds:  {seeds}")
    print(f"  Train:  {args.train_episodes} episodes")
    print(f"    (coaching K={K}, self-practice M={M} per iteration)")
    print(f"  Eval:   {args.eval_episodes} episodes (no coach)")
    print(f"  Device: {args.device}")
    if pi_path:
        print(f"  π_I:    {pi_path}")
    print(f"  π_A:    {bc_checkpoint}")
    print(f"  Output: {run_dir}")
    print("=" * 60)

    all_results = {}
    for mode in modes:
        mode_dir = run_dir / mode
        mode_dir.mkdir(exist_ok=True)

        results = []
        for seed in seeds:
            result = run_single_seed(
                mode=mode,
                seed=seed,
                pi_path=pi_path,
                bc_checkpoint=bc_checkpoint,
                run_dir=mode_dir,
                train_episodes=args.train_episodes,
                eval_episodes=args.eval_episodes,
                device=args.device,
            )
            results.append(result)

        summary = aggregate_results(results, mode_dir, mode)
        all_results[mode] = summary

    if len(modes) == 2:
        print(f"\n{'='*60}")
        print("  COMPARISON: With Coach vs No Coach")
        print(f"{'='*60}")
        for metric in ["success_rate_mean", "fall_rate_mean", "mean_reward_mean"]:
            wc = all_results["with_coach"]["evaluation_aggregate"][metric]
            nc = all_results["no_coach"]["evaluation_aggregate"][metric]
            label = metric.replace("_mean", "").replace("_", " ").title()
            if "rate" in metric:
                print(f"  {label:20s}: Coach={wc*100:.1f}%  vs  Solo={nc*100:.1f}%")
            else:
                print(f"  {label:20s}: Coach={wc:.2f}  vs  Solo={nc:.2f}")

        # Per-seed comparison
        print(f"\n  Per-seed eval success:")
        wc_per_seed = all_results["with_coach"]["evaluation_aggregate"]["success_rate_per_seed"]
        nc_per_seed = all_results["no_coach"]["evaluation_aggregate"]["success_rate_per_seed"]
        for s in seeds:
            wc_s = wc_per_seed.get(s, wc_per_seed.get(str(s), 0))
            nc_s = nc_per_seed.get(s, nc_per_seed.get(str(s), 0))
            diff = (wc_s - nc_s) * 100
            print(f"    seed {s}: Solo={nc_s*100:5.1f}%  Coach={wc_s*100:5.1f}%  "
                  f"Δ={diff:+5.1f}pp")
        print(f"{'='*60}")

    print(f"\n  Results saved to: {run_dir}")


if __name__ == "__main__":
    main()