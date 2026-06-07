# train/unicycle_train/coach/v5/train.py

"""
Coach V5 — Multi-Student Parallel Training with Trajectory Latent z

Identical to V4 from π_I's perspective (PPO, binary w_t, BC π_A, parallel students),
except that π_I observation is extended from [r, s] (2D) to [r, s, z] (34D),
where z (32D) is the latent representation of the student's last self-practice
episode trajectory, produced by a frozen autoencoder.

Outer loop unchanged from V4. Only changes:
  - obs_dim: 2 -> 34
  - Each worker loads a frozen encoder + dataset normalization stats
  - CoachWrapper computes z from the last self-practice trajectory inside get_pi_observation()

Usage:
    python -m train.unicycle_train.coach.v5.train
    python -m train.unicycle_train.coach.v5.train --config train/unicycle_train/coach/v5/configs/default.yaml
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

from train.unicycle_train.coach.v5.intervention_policy import InterventionPolicy
from train.unicycle_train.coach.v5.parallel_manager import ParallelManager


DEFAULT_CONFIG = {
    "experiment": {
        "name": "coach_v5",
    },

    "training": {
        "seed": 42,                     # coach/torch seed
        "K": 10,                        # coaching episodes per iteration per student
        "M": 10,                        # self-practice episodes per iteration per student
        "max_iterations": 300,          # same as V4 pool100
        "pi_update_every": 10,
        "pi_ppo_epochs": 4,
        "student_update_in_eval": True,

        "n_students": 100,
        "student_seeds_start": 1000,    # Pool B (matches V4): 1000-1099
    },

    "intervention_policy": {
        "obs_dim": 34,                  # V5: [r, s, z (32D)]
        "lr": 3e-4,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "entropy_coef": 0.01,
        "vf_coef": 0.5,
        "device": "cpu",
    },

    "assistance_policy": {
        "obs_dim": 22,
        "action_dim": 2,
        "device": "cpu",
    },

    # V5: frozen autoencoder for trajectory -> z
    "autoencoder": {
        "checkpoint": "train/unicycle_train/coach/v5/runs/ae_v5_baseline/best_model.pt",
        "dataset_stats": "train/unicycle_train/coach/v5/dataset_processed/train.npz",
        "device": "cpu",
    },

    "student": {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
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
    parser = argparse.ArgumentParser(description="Coach V5 Multi-Student Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override coach/torch seed")
    parser.add_argument("--bc_checkpoint", type=str, default=None,
                        help="Path to BC pretrained actor (auto-detects from V1 if omitted)")
    parser.add_argument("--ae_checkpoint", type=str, default=None,
                        help="Path to autoencoder checkpoint (overrides config)")
    parser.add_argument("--n_students", type=int, default=None,
                        help="Override number of parallel students")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.n_students is not None:
        config["training"]["n_students"] = args.n_students
    if args.ae_checkpoint is not None:
        config["autoencoder"]["checkpoint"] = args.ae_checkpoint

    SEED = config["training"]["seed"]
    sb3_set_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Set spawn start method once before anything else forks
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Directories
    script_dir = Path(__file__).resolve().parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    n_students = config["training"]["n_students"]
    run_dir = script_dir / "logs" / f"v5_{timestamp}_pool{n_students}"
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

    # V5: verify autoencoder checkpoint exists
    ae_checkpoint = Path(config["autoencoder"]["checkpoint"])
    if not ae_checkpoint.exists():
        print(f"[error] Autoencoder checkpoint not found: {ae_checkpoint}")
        exit(1)
    (run_dir / "ae_checkpoint.txt").write_text(str(ae_checkpoint))

    # Student seeds: consecutive, starting from student_seeds_start (Pool B)
    seed_start = config["training"]["student_seeds_start"]
    student_seeds = list(range(seed_start, seed_start + n_students))

    # Initialize π_I on main process
    intervention_policy = InterventionPolicy(
        obs_dim=config["intervention_policy"]["obs_dim"],
        lr=config["intervention_policy"]["lr"],
        gamma=config["intervention_policy"]["gamma"],
        clip_eps=config["intervention_policy"]["clip_eps"],
        entropy_coef=config["intervention_policy"]["entropy_coef"],
        vf_coef=config["intervention_policy"]["vf_coef"],
        device=config["intervention_policy"]["device"],
    )
    print(f"[INIT] π_I initialized (PPO, obs_dim={config['intervention_policy']['obs_dim']}, action={{0,1}})")

    K = config["training"]["K"]
    M = config["training"]["M"]
    max_iterations = config["training"]["max_iterations"]
    update_every = config["training"]["pi_update_every"]
    student_update_in_eval = config["training"]["student_update_in_eval"]

    print()
    print("=" * 70)
    print("[INFO] Coach V5 — Multi-Student Parallel Training with Trajectory Latent")
    print("=" * 70)
    print(f"  Coach/torch seed:       {SEED}")
    print(f"  n_students (workers):   {n_students}")
    print(f"  Student seeds (Pool B): {student_seeds[0]}..{student_seeds[-1]}")
    print(f"  K (coaching):           {K} episodes/iteration/student")
    print(f"  M (self-practice):      {M} episodes/iteration/student")
    print(f"  Max iterations:         {max_iterations}")
    print(f"  π_I obs_dim:            {config['intervention_policy']['obs_dim']} ([r, s, z (32D)])")
    print(f"  π_I update every:       {update_every} iterations")
    print(f"  Student update in eval: {student_update_in_eval}")
    print(f"  BC checkpoint (π_A):    {bc_checkpoint}")
    print(f"  AE checkpoint (z):      {ae_checkpoint}")
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
        for iter_idx in range(1, max_iterations + 1):
            iter_start = time.time()

            # Step 1: π_I predicts w_t for each worker (batched)
            w_ts, log_probs, values = intervention_policy.predict_batch(
                s_obs, deterministic=False,
            )

            # Step 2: dispatch + wait (barrier sync)
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

                # Store transition for this worker's (s_obs, w_t, reward)
                intervention_policy.store_transition(
                    obs=s_obs[i],
                    action=w_ts[i],
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
            intervention_rate = float(np.mean(w_ts))

            # V5: track z statistics (last 32 dims of s_obs)
            z_mean = float(new_s_obs[:, 2:].mean())
            z_std = float(new_s_obs[:, 2:].std())

            log_entry = {
                "iteration": iter_idx,
                "w_ts": [int(w) for w in w_ts],
                "intervention_rate": intervention_rate,
                "coach_rewards_per_worker": coach_rewards,
                "mean_coach_reward": mean_coach_reward,
                "mean_coaching_success": mean_coaching_success,
                "mean_coaching_reward": mean_coaching_reward,
                "mean_eval_success": mean_eval_success,
                "mean_eval_reward": mean_eval_reward,
                "mean_coach_action_norm": float(np.mean(coach_action_norms)),
                "z_mean": z_mean,
                "z_std": z_std,
                "total_episodes": total_episodes,
                "total_steps": total_steps,
                "wall_time": iter_elapsed,
                **pi_logs,
            }
            training_log.append(log_entry)

            # Tensorboard
            writer.add_scalar("coach/intervention_rate", intervention_rate, iter_idx)
            writer.add_scalar("coach/mean_coach_reward", mean_coach_reward, iter_idx)
            writer.add_scalar("coaching/mean_avg_reward", mean_coaching_reward, iter_idx)
            writer.add_scalar("coaching/mean_success_rate", mean_coaching_success, iter_idx)
            writer.add_scalar("eval/mean_avg_reward", mean_eval_reward, iter_idx)
            writer.add_scalar("eval/mean_success_rate", mean_eval_success, iter_idx)
            writer.add_scalar("coaching/mean_coach_action_norm",
                              float(np.mean(coach_action_norms)), iter_idx)
            writer.add_scalar("z/mean", z_mean, iter_idx)
            writer.add_scalar("z/std", z_std, iter_idx)
            writer.add_scalar("training/total_episodes", total_episodes, iter_idx)
            writer.add_scalar("training/total_steps", total_steps, iter_idx)
            writer.add_scalar("training/iter_wall_time", iter_elapsed, iter_idx)
            for k, v in pi_logs.items():
                writer.add_scalar(f"pi_I/{k}", v, iter_idx)
            writer.flush()

            # Console
            print(
                f"[Iter {iter_idx:4d}] "
                f"intervene={intervention_rate*100:4.0f}% | "
                f"coaching: r={mean_coaching_reward:+7.1f} s={mean_coaching_success*100:4.0f}% | "
                f"eval: r={mean_eval_reward:+7.1f} s={mean_eval_success*100:4.0f}% | "
                f"z(std)={z_std:.2f} | "
                f"eps={total_episodes:6d} | "
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