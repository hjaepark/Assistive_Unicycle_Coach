# train/unicycle_train/coach/v3_2/eval.py

"""
Coach V3.2 Evaluation

Evaluates the trained V3.2 coach (π_I + π_A) by:
  Phase A: Train a NEW student with frozen coach for TRAIN_EPISODES
           using observation phase followed by coaching → self-practice cycle
           (matching train.py V3.2 structure)
  Phase B: Remove coach, evaluate student alone for EVAL_EPISODES

V3.2 difference from V3:
  - Observation phase (first N_OBSERVATION_ITERS iterations):
      w_t=0 forced, only self-practice runs
      π_I passively observes student performance
      π_obs ([avg_reward, success_rate]) is updated from self-practice
  - Intervention phase (remaining iterations):
      Standard V3.1 logic (π_I decides w_t, coaching → self-practice cycle)

Runs both "with_coach" and "no_coach" modes for direct comparison.
Multiple seeds supported for statistical significance.

Usage:
    # Both modes (with coach + no coach baseline):
    python -m train.unicycle_train.coach.v3_2.eval --pi-path <path_to_pi_I.pth> --seeds 42 43 44 45 46

    # With coach only:
    python -m train.unicycle_train.coach.v3_2.eval --mode with_coach --pi-path <path> --seed 42

    # No coach baseline only:
    python -m train.unicycle_train.coach.v3_2.eval --mode no_coach --seed 42

    # Custom observation phase length (ablation):
    python -m train.unicycle_train.coach.v3_2.eval --pi-path <path> --seed 42 --n-observation-iters 5

    # Custom episode counts:
    python -m train.unicycle_train.coach.v3_2.eval --pi-path <path> --train-episodes 1000 --eval-episodes 200
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

# V3.2 reuses V3 components (intervention_policy, wrapper)
from train.unicycle_train.coach.v0.coach_agent import CoachAgent
from train.unicycle_train.coach.v3.intervention_policy import InterventionPolicy
from train.unicycle_train.coach.v3.wrapper import CoachWrapper


TRAIN_EPISODES = 500
EVAL_EPISODES = 100
K = 10
M = 10
N_OBSERVATION_ITERS = 20  # V3.2: number of observation-phase iterations

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

def make_env_with_v3_coach(log_dir, seed, assistance_policy, intervention_policy):
    base_env = gym.make(
        "UnicyclePendulumPD-v0",
        reward_weights=REWARD_WEIGHTS,
    )
    monitored_env = Monitor(base_env, filename=str(log_dir / "monitor.csv"))

    # V3 wrapper with both policies (reused for V3.2)
    wrapped_env = CoachWrapper(
        env=monitored_env,
        assistance_policy=assistance_policy,
        w_t=0,  # Will be set by π_I each iteration
    )
    wrapped_env.reset(seed=seed)
    wrapped_env.action_space.seed(seed)
    wrapped_env.observation_space.seed(seed)
    return wrapped_env

def load_assistance_policy(bc_checkpoint, device="cuda"):
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

def load_intervention_policy(pi_path, device="cuda"):
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

def train_with_v3_2_coach(student, env, intervention_policy, train_episodes,
                          n_observation_iters=N_OBSERVATION_ITERS,
                          coaching_K=K, practice_M=M):
    """
    V3.2 training procedure:
      - First n_observation_iters iterations: observation phase (w_t=0 forced,
        only self-practice runs, π_obs updated from self-practice outcomes)
      - Remaining iterations: intervention phase (standard V3.1 logic)
    """
    all_logs = []
    episodes_done = 0
    pi_obs = np.array([0.0, 0.0], dtype=np.float32)
    iter_idx = 0

    while episodes_done < train_episodes:
        # ===== V3.2: Observation Phase =====
        if iter_idx < n_observation_iters:
            # Force w_t=0, skip coaching, run self-practice only
            w_t = 0
            env.set_intervention(w_t)

            env.set_mode("eval")  # self-practice mode
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

            # Log self-practice entries with observation-phase tag
            for entry in logger_sp.episode_logs:
                entry["w_t"] = -1  # No coach
                entry["phase"] = "observation"
                all_logs.append(entry)

            episodes_done += stop_cb_sp.episode_count

            # Update π_I observation from self-practice
            # (π_I passively observes student performance during observation phase)
            pi_obs = env.get_pi_observation()

            iter_idx += 1
            continue

        # ===== V3.2: Intervention Phase (standard V3.1 logic) =====
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

        # Update π_I observation from coaching summary
        pi_obs = env.get_pi_observation()

        if episodes_done >= train_episodes:
            break

        env.set_mode("eval")
        remaining = train_episodes - episodes_done
        chunk_M = min(practice_M, remaining)

        stop_cb_sp = StopAfterEpisodes(max_episodes=chunk_M)
        logger_sp = EpisodeLogger()

        # Sync SB3 state
        obs, _ = env.reset()
        if student._last_obs is not None:
            student._last_obs = np.array([obs])

        student.learn(
            total_timesteps=10_000_000,
            callback=[stop_cb_sp, logger_sp],
            reset_num_timesteps=False,
            progress_bar=False,
        )

        # Collect self-practice logs
        for entry in logger_sp.episode_logs:
            entry["w_t"] = -1  # No coach
            entry["phase"] = "self_practice"
            all_logs.append(entry)

        episodes_done += stop_cb_sp.episode_count

        iter_idx += 1

    return all_logs

def evaluate_independent(student, eval_episodes, seed, eval_dir, device="cuda"):
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
                    train_episodes, eval_episodes,
                    n_observation_iters=N_OBSERVATION_ITERS,
                    device="cuda"):
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
    if mode == "with_coach":
        print(f"  Observation iters: {n_observation_iters}")
    print(f"{'='*60}")

    print(f"\n  [Phase A] Training ({train_episodes} episodes)...")
    train_start = time.time()

    if mode == "with_coach":
        # Load frozen coach components
        assistance_policy = load_assistance_policy(bc_checkpoint, device=device)
        pi_I = load_intervention_policy(pi_path, device=device)

        train_env = make_env_with_v3_coach(
            train_dir, seed, assistance_policy, pi_I
        )

        student = SAC(
            "MlpPolicy",
            train_env,
            **STUDENT_CONFIG,
            device=device,
            verbose=0,
            seed=seed,
        )

        # Train with V3.2 coach: observation phase + intervention phase
        train_logs = train_with_v3_2_coach(
            student, train_env, pi_I, train_episodes,
            n_observation_iters=n_observation_iters
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

    # For V3.2 with_coach: report observation-phase vs intervention-phase breakdown
    if mode == "with_coach":
        obs_logs = [e for e in train_logs if e.get("phase") == "observation"]
        int_coaching_logs = [e for e in train_logs if e.get("phase") == "coaching"]
        int_sp_logs = [e for e in train_logs if e.get("phase") == "self_practice"]
        if obs_logs:
            obs_s = sum(1 for e in obs_logs if e["goal_reached"])
            print(f"    [Observation phase]   {len(obs_logs)} eps, "
                  f"{obs_s}/{len(obs_logs)} success "
                  f"({100*obs_s/max(1,len(obs_logs)):.1f}%)")
        if int_coaching_logs or int_sp_logs:
            ic_s = sum(1 for e in int_coaching_logs if e["goal_reached"])
            is_s = sum(1 for e in int_sp_logs if e["goal_reached"])
            print(f"    [Intervention phase]  coaching: {len(int_coaching_logs)} eps, "
                  f"{ic_s}/{len(int_coaching_logs)} success "
                  f"({100*ic_s/max(1,len(int_coaching_logs)):.1f}%)")
            print(f"                          self-practice: {len(int_sp_logs)} eps, "
                  f"{is_s}/{len(int_sp_logs)} success "
                  f"({100*is_s/max(1,len(int_sp_logs)):.1f}%)")

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
        "n_observation_iters": n_observation_iters if mode == "with_coach" else None,

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
        "n_observation_iters": results[0].get("n_observation_iters"),

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
    print(f"{'='*60}")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Coach V3.2 Evaluation")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["no_coach", "with_coach", "both"],
                        help="Evaluation mode (default: both)")
    parser.add_argument("--pi-path", type=str, default=None,
                        help="Path to trained π_I checkpoint (.pth)")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Path to BC pretrained actor (auto-detects from V1 if omitted)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds to run")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for training (default: cuda)")
    parser.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES,
                        help=f"Training episodes (default: {TRAIN_EPISODES})")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES,
                        help=f"Evaluation episodes (default: {EVAL_EPISODES})")
    parser.add_argument("--n-observation-iters", type=int, default=N_OBSERVATION_ITERS,
                        help=f"V3.2 observation phase iterations (default: {N_OBSERVATION_ITERS}). "
                             f"Use 0 to disable observation phase.")
    args = parser.parse_args()

    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42, 43, 44, 45, 46]

    # Find BC checkpoint
    script_dir = Path(__file__).resolve().parent
    if args.bc_checkpoint is not None:
        bc_checkpoint = Path(args.bc_checkpoint)
    else:
        v1_dir = script_dir.parent / "v1"
        bc_checkpoint = find_latest_bc_checkpoint(v1_dir)

    # Find π_I checkpoint
    pi_path = None
    if args.mode in ("with_coach", "both"):
        if args.pi_path is None:
            # Auto-detect latest V3.2 run
            logs_dir = script_dir / "logs"
            v3_2_runs = sorted(logs_dir.glob("v3_2_*"))
            if not v3_2_runs:
                parser.error("No V3.2 runs found. Provide --pi-path explicitly.")
            pi_path = v3_2_runs[-1] / "models" / "pi_I_final.pth"
        else:
            pi_path = Path(args.pi_path)

        if not pi_path.exists():
            parser.error(f"π_I checkpoint not found: {pi_path}")

    # Determine which modes to run
    modes = []
    if args.mode == "both":
        modes = ["no_coach", "with_coach"]
    else:
        modes = [args.mode]

    # Create run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if len(seeds) == 1:
        seed_tag = f"seed{seeds[0]}"
    else:
        seed_tag = f"seeds{'_'.join(map(str, seeds))}"
    # Include observation iters in tag when running with_coach
    obs_tag = f"_obs{args.n_observation_iters}" if "with_coach" in modes else ""
    run_dir = script_dir / "eval_logs" / f"eval_{timestamp}_{seed_tag}{obs_tag}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Save eval config
    eval_config = {
        "modes": modes,
        "pi_path": str(pi_path) if pi_path else None,
        "bc_checkpoint": str(bc_checkpoint),
        "seeds": seeds,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "coaching_K": K,
        "practice_M": M,
        "n_observation_iters": args.n_observation_iters,
        "reward_weights": REWARD_WEIGHTS,
        "student_config": STUDENT_CONFIG,
        "device": args.device,
    }
    with open(run_dir / "eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)

    print("=" * 60)
    print("  Coach V3.2 Evaluation")
    print("=" * 60)
    print(f"  Modes:  {modes}")
    print(f"  Seeds:  {seeds}")
    print(f"  Train:  {args.train_episodes} episodes")
    print(f"    (observation iters={args.n_observation_iters}, "
          f"coaching K={K}, self-practice M={M} per iteration)")
    print(f"  Eval:   {args.eval_episodes} episodes (no coach)")
    print(f"  Device: {args.device}")
    if pi_path:
        print(f"  π_I:    {pi_path}")
    print(f"  π_A:    {bc_checkpoint}")
    print(f"  Output: {run_dir}")
    print("=" * 60)

    # Run all modes and seeds
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
                n_observation_iters=args.n_observation_iters,
                device=args.device,
            )
            results.append(result)

        summary = aggregate_results(results, mode_dir, mode)
        all_results[mode] = summary

    # Print comparison if both modes ran
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
        print(f"{'='*60}")

    print(f"\n  Results saved to: {run_dir}")


if __name__ == "__main__":
    main()