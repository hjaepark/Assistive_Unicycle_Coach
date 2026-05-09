# train/unicycle_train/coach/v4/train.py

"""
Coach V4 — Multi-Student Parallel Training

Identical to V3.1 from π_I's perspective (PPO, 2D obs, binary w_t, BC π_A),
except that training experience now comes from N students in parallel
instead of one student.

Outer loop:
  For each iteration:
    1. Collect s_obs from all N workers (last self-practice summary)
    2. π_I.predict_batch(s_obs) -> N w_t values
    3. Send w_t to each worker; workers run K coaching + M self-practice in parallel
    4. Collect (s_obs_next, coach_reward) from each worker
    5. Store N transitions in π_I's buffer
    6. Every pi_update_every iterations, run PPO update on π_I

Architecture unchanged from V3.1. Only training regime differs.

Usage:
    python -m train.unicycle_train.coach.v4_2.train
    python -m train.unicycle_train.coach.v4_2.train --config train/unicycle_train/coach/v4/configs/default.yaml
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from stable_baselines3.common.utils import set_random_seed as sb3_set_seed

from train.unicycle_train.coach.v4_2.intervention_policy import InterventionPolicy
from train.unicycle_train.coach.v4_2.parallel_manager import ParallelManager


DEFAULT_CONFIG = {
    "experiment": {
        "name": "coach_v4",
    },

    "training": {
        "seed": 42,                     # coach/torch seed (reproducibility of π_I only)
        "K": 10,                        # coaching episodes per iteration per student
        "M": 10,                        # self-practice episodes per iteration per student
        "max_iterations": 500,          # total outer iterations (same as V3.1)
        "pi_update_every": 10,          # π_I PPO update every N iterations
        "pi_ppo_epochs": 4,
        "student_update_in_eval": True, # V3.1 setting

        # [V4 ADDITION] Multi-student settings
        "n_students": 10,
        "student_seeds_start": 1000,    # student seeds: start, start+1, ..., start+n-1
    },

    "intervention_policy": {
        "obs_dim": 2,
        "n_actions": 5,                              # V4.1 NEW: 5 graduated levels
        "w_values": [0.0, 0.25, 0.5, 0.75, 1.0],     # V4.1 NEW: explicit mapping
        "lr": 3e-4,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "entropy_coef": 0.02,                        # V4.1: raised from 0.01 for action exploration
        "vf_coef": 0.5,
        "device": "cpu",
    },

    "assistance_policy": {
        "obs_dim": 22,
        "action_dim": 2,
        "device": "cpu",
    },

    "student": {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,       # same as V3.1
        "learning_starts": 1_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
    },

    "reward_weights": {
        "posture": 0.0,
        "pendulum": 0.0,
        "speed_reward_weight": 0.0,
        "alive_bonus": 0.0,
        "upright_gate_k": 0.0,
        "progress": 60.0,
        "goal_bonus": 50.0,
        "fall_penalty": -20.0,
    },
}


def load_config(config_path=None):
    config = {k: v.copy() if isinstance(v, dict) else v for k, v in DEFAULT_CONFIG.items()}
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
    """Find BC checkpoint in V1 logs (reuse V1's pretrained π_A)."""
    logs_dir = v1_dir / "logs"
    bc_runs = sorted(logs_dir.glob("bc_*"))
    if not bc_runs:
        raise FileNotFoundError(f"No BC runs found in {logs_dir}")
    checkpoint = bc_runs[-1] / "checkpoints" / "best_bc_actor.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No best_bc_actor.pth in {bc_runs[-1]}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Coach V4 Multi-Student Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override coach/torch seed")
    parser.add_argument("--bc_checkpoint", type=str, default=None,
                        help="Path to BC pretrained actor (auto-detects from V1 if omitted)")
    parser.add_argument("--n_students", type=int, default=None,
                        help="Override number of parallel students")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.n_students is not None:
        config["training"]["n_students"] = args.n_students

    SEED = config["training"]["seed"]
    sb3_set_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # [V4 REQUIREMENT] Set spawn start method once before anything else forks
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set
        pass

    # Directories
    script_dir = Path(__file__).resolve().parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    n_students = config["training"]["n_students"]
    run_dir = script_dir / "logs" / f"v4_1_{timestamp}_pool{n_students}"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    workers_dir = run_dir / "workers"
    workers_dir.mkdir(exist_ok=True)

    with open(run_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    (run_dir / "seed.txt").write_text(f"{SEED}\n")

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    # BC checkpoint (for π_A)
    if args.bc_checkpoint is not None:
        bc_checkpoint = Path(args.bc_checkpoint)
    else:
        v1_dir = script_dir.parent / "v1"
        bc_checkpoint = find_latest_bc_checkpoint(v1_dir)

    if not bc_checkpoint.exists():
        print(f"[error] BC checkpoint not found: {bc_checkpoint}")
        exit(1)

    (run_dir / "bc_checkpoint.txt").write_text(str(bc_checkpoint))

    # Student seeds: consecutive, starting from student_seeds_start
    seed_start = config["training"]["student_seeds_start"]
    student_seeds = list(range(seed_start, seed_start + n_students))

    # Initialize π_I on main process
    intervention_policy = InterventionPolicy(
        obs_dim=config["intervention_policy"]["obs_dim"],
        n_actions=config["intervention_policy"]["n_actions"],         # V4.1 NEW
        w_values=config["intervention_policy"]["w_values"],           # V4.1 NEW
        lr=config["intervention_policy"]["lr"],
        gamma=config["intervention_policy"]["gamma"],
        clip_eps=config["intervention_policy"]["clip_eps"],
        entropy_coef=config["intervention_policy"]["entropy_coef"],
        vf_coef=config["intervention_policy"]["vf_coef"],
        device=config["intervention_policy"]["device"],
    )
    print(f"[INIT] π_I initialized (PPO, obs_dim=2, n_actions={intervention_policy.n_actions}, "
        f"w_values={intervention_policy.w_values})")

    K = config["training"]["K"]
    M = config["training"]["M"]
    max_iterations = config["training"]["max_iterations"]
    update_every = config["training"]["pi_update_every"]
    student_update_in_eval = config["training"]["student_update_in_eval"]

    print()
    print("=" * 70)
    print("[INFO] Coach V4 — Multi-Student Parallel Training")
    print("=" * 70)
    print(f"  Coach/torch seed:       {SEED}")
    print(f"  n_students (workers):   {n_students}")
    print(f"  Student seeds:          {student_seeds[0]}..{student_seeds[-1]}")
    print(f"  K (coaching):           {K} episodes/iteration/student")
    print(f"  M (self-practice):      {M} episodes/iteration/student")
    print(f"  Max iterations:         {max_iterations}")
    print(f"  π_I update every:       {update_every} iterations")
    print(f"  Student update in eval: {student_update_in_eval}")
    print(f"  BC checkpoint (π_A):    {bc_checkpoint}")
    print(f"  Run dir:                {run_dir}")
    print("=" * 70)

    # Spawn workers
    manager = ParallelManager(
        n_workers=n_students,
        student_seeds=student_seeds,
        bc_checkpoint=bc_checkpoint,
        config=config,
        run_dir=workers_dir,
    )
    manager.start()

    training_log = []

    # π_I's current observation for each worker. Initialized to zeros.
    s_obs = np.zeros((n_students, config["intervention_policy"]["obs_dim"]), dtype=np.float32)

    total_episodes = 0
    total_steps = 0

    try:
        # Inside the main loop in train.py, replace the existing block

        for iter_idx in range(1, max_iterations + 1):
            iter_start = time.time()

            # Step 1: π_I predicts for each worker (batched)
            # V4.1 CHANGE: now unpacking 4 return values instead of 3
            # action_idxs are discrete indices for buffer/PPO; w_ts are floats for wrappers
            action_idxs, w_ts, log_probs, values = intervention_policy.predict_batch(
                s_obs, deterministic=False,
            )

            # Step 2: dispatch + wait (barrier sync)
            # V4.1 CHANGE: send w_ts (floats) to workers, not action_idxs
            # The wrapper inside each worker now interprets w_t as a continuous scaling factor
            results = manager.step_all(w_ts)

            # Step 3: gather results and store N transitions
            new_s_obs = np.zeros_like(s_obs)
            coach_rewards = []
            coaching_success_rates = []
            coaching_avg_rewards = []
            eval_success_rates = []
            eval_avg_rewards = []
            coach_action_norms = []

            for i, res in enumerate(results):
                new_s_obs[i] = res["s_obs_next"]
                coach_rewards.append(res["coach_reward"])
                coaching_success_rates.append(res["coaching_summary"]["success_rate"])
                coaching_avg_rewards.append(res["coaching_summary"]["avg_reward"])
                eval_success_rates.append(res["eval_summary"]["success_rate"])
                eval_avg_rewards.append(res["eval_summary"]["avg_reward"])
                coach_action_norms.append(res["coaching_summary"]["avg_coach_action_norm"])

                total_episodes += res["n_episodes"]
                total_steps += res["n_steps"]

                # V4.1 CHANGE: store action_idx (integer) in buffer, not w_t (float)
                # PPO's categorical log_prob is defined on the index space
                intervention_policy.store_transition(
                    obs=s_obs[i],
                    action=action_idxs[i],     # integer index, NOT the float w_t
                    log_prob=log_probs[i],
                    reward=res["coach_reward"],
                    value=values[i],
                    done=False,
                )

            # Update s_obs for next iteration
            s_obs = new_s_obs

            # Step 4: periodic PPO update
            pi_logs = {}
            if iter_idx % update_every == 0:
                pi_logs = intervention_policy.train_step(
                    n_epochs=config["training"]["pi_ppo_epochs"]
                )

            iter_elapsed = time.time() - iter_start

            # Aggregate per-iteration metrics across workers
            mean_coach_reward = float(np.mean(coach_rewards))
            mean_coaching_success = float(np.mean(coaching_success_rates))
            mean_coaching_reward = float(np.mean(coaching_avg_rewards))
            mean_eval_success = float(np.mean(eval_success_rates))
            mean_eval_reward = float(np.mean(eval_avg_rewards))

            # V4.1 CHANGE: split intervention metric into rate and intensity
            # Rate is the fraction of workers receiving any nonzero coaching this iteration
            # Intensity is the average magnitude of w_t across all workers
            # In V4 these were the same number; in V4.1 they diverge
            intervention_rate = float(np.mean([abs(w) > 1e-8 for w in w_ts]))
            intervention_intensity = float(np.mean(w_ts))

            # Also useful: distribution of which actions were chosen
            action_counts = np.bincount(action_idxs, minlength=intervention_policy.n_actions)
            action_distribution = (action_counts / len(action_idxs)).tolist()

            log_entry = {
                "iteration": iter_idx,
                "w_ts": [float(w) for w in w_ts],   # V4.1: keep float precision for analysis
                "action_idxs": list(action_idxs),   # V4.1 NEW: also log discrete indices for distribution analysis
                "intervention_rate": intervention_rate,
                "coach_rewards_per_worker": coach_rewards,
                "mean_coach_reward": mean_coach_reward,
                "mean_coaching_success": mean_coaching_success,
                "mean_coaching_reward": mean_coaching_reward,
                "mean_eval_success": mean_eval_success,
                "mean_eval_reward": mean_eval_reward,
                "mean_coach_action_norm": float(np.mean(coach_action_norms)),
                "total_episodes": total_episodes,
                "total_steps": total_steps,
                "wall_time": iter_elapsed,
                **pi_logs,
            }
            training_log.append(log_entry)

            # Tensorboard
            writer.add_scalar("coach/intervention_rate", intervention_rate, iter_idx)
            writer.add_scalar("coach/intervention_intensity", intervention_intensity, iter_idx)
            # Log action distribution as separate scalars (one per action) so they show up as a grouped chart
            for j, frac in enumerate(action_distribution):
                w_label = intervention_policy.w_values[j]
                writer.add_scalar(f"coach/action_freq/w_{w_label:.2f}", frac, iter_idx)
            writer.add_scalar("coach/mean_coach_reward", mean_coach_reward, iter_idx)
            writer.add_scalar("coaching/mean_avg_reward", mean_coaching_reward, iter_idx)
            writer.add_scalar("coaching/mean_success_rate", mean_coaching_success, iter_idx)
            writer.add_scalar("eval/mean_avg_reward", mean_eval_reward, iter_idx)
            writer.add_scalar("eval/mean_success_rate", mean_eval_success, iter_idx)
            writer.add_scalar("coaching/mean_coach_action_norm",
                              float(np.mean(coach_action_norms)), iter_idx)
            writer.add_scalar("training/total_episodes", total_episodes, iter_idx)
            writer.add_scalar("training/total_steps", total_steps, iter_idx)
            writer.add_scalar("training/iter_wall_time", iter_elapsed, iter_idx)
            for k, v in pi_logs.items():
                writer.add_scalar(f"pi_I/{k}", v, iter_idx)
            writer.flush()

            # Console
            print(
                f"[Iter {iter_idx:4d}] "
                f"intervene={intervention_rate*100:4.0f}% intensity={intervention_intensity:.2f} | "
                f"coaching: r={mean_coaching_reward:+7.1f} s={mean_coaching_success*100:4.0f}% | "
                f"eval: r={mean_eval_reward:+7.1f} s={mean_eval_success*100:4.0f}% | "
                f"eps={total_episodes:6d} steps={total_steps:8d} | "
                f"{iter_elapsed:.1f}s"
            )

            # Periodic saves
            if iter_idx % 10 == 0:
                with open(run_dir / "training_log.json", 'w') as f:
                    json.dump(training_log, f, indent=2)

            if iter_idx % 50 == 0:
                intervention_policy.save(models_dir / f"pi_I_iter_{iter_idx:04d}.pth")

    except KeyboardInterrupt:
        print("\n\n[INFO] Training stopped by user (Ctrl+C)")

    except Exception as e:
        print(f"\n[ERROR] Training crashed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Save π_I
        intervention_policy.save(models_dir / "pi_I_final.pth")
        print(f"\n[INFO] π_I saved: {models_dir / 'pi_I_final.pth'}")

        # Save log
        with open(run_dir / "training_log.json", 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"[INFO] Log saved: {run_dir / 'training_log.json'}")

        # Close workers (saves final students)
        try:
            manager.close(save_dir=models_dir / "students")
        except Exception as e:
            print(f"[WARN] Error during manager.close(): {e}")

        writer.close()

        if training_log:
            intervention_rates = [e["intervention_rate"] for e in training_log]
            mean_rewards = [e["mean_coach_reward"] for e in training_log]
            eval_successes = [e["mean_eval_success"] for e in training_log]

            print(f"\n{'='*70}")
            print(f"  Training Summary ({len(training_log)} iterations)")
            print(f"{'='*70}")
            print(f"  Pool size:         {n_students} students (seeds "
                  f"{student_seeds[0]}..{student_seeds[-1]})")
            print(f"  Total episodes:    {total_episodes}")
            print(f"  Total steps:       {total_steps}")
            print(f"  Mean intervention rate: {np.mean(intervention_rates)*100:.1f}%")
            print(f"  Last 10 intervention:   {np.mean(intervention_rates[-10:])*100:.1f}%")
            print(f"  Mean coach reward:      {np.mean(mean_rewards):.2f}")
            print(f"  Last 10 coach reward:   {np.mean(mean_rewards[-10:]):.2f}")
            print(f"  Mean eval success:      {np.mean(eval_successes)*100:.1f}%")
            print(f"  Last 10 eval success:   {np.mean(eval_successes[-10:])*100:.1f}%")
            print(f"{'='*70}")


if __name__ == "__main__":
    main()