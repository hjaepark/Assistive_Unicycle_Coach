# train/unicycle_train/coach/v1/build_dataset.py

"""
Coach V1 - Offline Dataset Builder

Generates a BC (Behavior Cloning) dataset for coach pretraining.
For each qualified student run:
  1. Load trajectory.csv to extract novice-phase (state, action, reward)
  2. Load best_student_model.zip as expert policy
  3. Compute delta = expert_action - novice_action for each state
  4. Build coach_inputs (22D) matching V0 CoachAgent obs_dim
  5. Save dataset as .npz

Usage:
    python -m train.unicycle_train.coach.v1.build_dataset
    python -m train.unicycle_train.coach.v1.build_dataset --min_eval_success 0.95
    python -m train.unicycle_train.coach.v1.build_dataset --novice_ratio 0.3 --dry_run
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import gymnasium as gym
from stable_baselines3 import SAC

import envs.unicycle

DEFAULT_CONFIG = {
    "min_eval_success_rate": 0.80,
    "novice_ratio": 0.25,
    "skip_initial_steps": 5,
    "subsample_ratio": 1.0,
    "env_id": "UnicyclePendulumPD-v0",
    "obs_dim": 19,
    "action_dim": 2,
}

def load_eval_results(run_dir: Path) -> Optional[Dict]:
    eval_path = run_dir / "evaluation_results.json"
    if not eval_path.exists():
        return None
    with open(eval_path, 'r') as f:
        return json.load(f)


def load_seed(run_dir: Path) -> Optional[str]:
    seed_path = run_dir / "seed.txt"
    if not seed_path.exists():
        return None
    return seed_path.read_text().strip()


def load_trajectory(run_dir: Path) -> Optional[pd.DataFrame]:
    traj_path = run_dir / "trajectory.csv"
    if not traj_path.exists():
        return None
    return pd.read_csv(traj_path)


def load_monitor(run_dir: Path) -> Optional[pd.DataFrame]:
    monitor_path = run_dir / "monitor.csv"
    if not monitor_path.exists():
        return None
    with open(monitor_path, 'r') as f:
        first_line = f.readline()
    skiprows = 1 if first_line.startswith('#') else 0
    return pd.read_csv(monitor_path, skiprows=skiprows)


def load_expert_model(run_dir: Path, env_id: str) -> Optional[SAC]:
    model_path = run_dir / "models" / "best_student_model.zip"
    if not model_path.exists():
        model_path = run_dir / "models" / "student_agent.zip"
    if not model_path.exists():
        return None
    try:
        model = SAC.load(model_path.as_posix(), device="cpu")
        return model
    except Exception as e:
        print(f"  [WARN] Failed to load model from {model_path}: {e}")
        return None


def classify_episodes_by_reward(monitor_df: pd.DataFrame, novice_ratio: float) -> Tuple[set, set]:
    total_episodes = len(monitor_df)
    novice_cutoff = int(total_episodes * novice_ratio)
    novice_episodes = set(range(1, novice_cutoff + 1))
    expert_episodes = set(range(novice_cutoff + 1, total_episodes + 1))
    return novice_episodes, expert_episodes


def extract_novice_data(
    traj_df: pd.DataFrame,
    novice_episodes: set,
    obs_dim: int,
    skip_initial_steps: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (state, novice_action, reward) from novice episodes.
    """
    obs_cols = [f"obs_{i}" for i in range(obs_dim)]
    action_cols = ["human_action_0", "human_action_1"]

    novice_mask = traj_df["episode"].isin(novice_episodes)
    novice_df = traj_df[novice_mask].copy()

    if skip_initial_steps > 0:
        novice_df["ep_step"] = novice_df.groupby("episode").cumcount()
        novice_df = novice_df[novice_df["ep_step"] >= skip_initial_steps]

    states = novice_df[obs_cols].values.astype(np.float32)
    actions = novice_df[action_cols].values.astype(np.float32)
    rewards = novice_df["reward"].values.astype(np.float32)

    return states, actions, rewards


def compute_expert_actions(
    model: SAC,
    states: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    expert_actions = []
    n = len(states)
    for i in range(0, n, batch_size):
        batch = states[i:i + batch_size]
        actions, _ = model.predict(batch, deterministic=True)
        expert_actions.append(actions)
    return np.concatenate(expert_actions, axis=0).astype(np.float32)


def process_single_student(
    run_dir: Path,
    config: Dict,
) -> Optional[Dict[str, np.ndarray]]:
    run_name = run_dir.name
    seed = load_seed(run_dir)

    eval_results = load_eval_results(run_dir)
    if eval_results is None:
        print(f"  [SKIP] {run_name}: no evaluation_results.json")
        return None

    success_rate = eval_results.get("current_success_rate", 0.0)
    if success_rate < config["min_eval_success_rate"]:
        print(f"  [SKIP] {run_name} (seed={seed}): eval success rate {success_rate:.2f} < {config['min_eval_success_rate']:.2f}")
        return None

    traj_df = load_trajectory(run_dir)
    if traj_df is None or len(traj_df) == 0:
        print(f"  [SKIP] {run_name}: no trajectory data")
        return None

    monitor_df = load_monitor(run_dir)
    if monitor_df is None:
        print(f"  [SKIP] {run_name}: no monitor.csv")
        return None

    novice_episodes, expert_episodes = classify_episodes_by_reward(
        monitor_df, config["novice_ratio"]
    )

    states, novice_actions, rewards = extract_novice_data(
        traj_df, novice_episodes, config["obs_dim"], config["skip_initial_steps"]
    )

    if len(states) == 0:
        print(f"  [SKIP] {run_name}: no novice data after filtering")
        return None

    expert_model = load_expert_model(run_dir, config["env_id"])
    if expert_model is None:
        print(f"  [SKIP] {run_name}: could not load expert model")
        return None

    expert_actions = compute_expert_actions(expert_model, states)
    deltas = expert_actions - novice_actions

    # Coach input (22D): [state(19) + student_action(2) + reward(1)]
    # Matches V0 CoachAgent obs_dim=22 for direct weight transfer
    coach_inputs = np.concatenate([
        states,                              # (N, 19)
        novice_actions,                      # (N, 2)
        rewards.reshape(-1, 1),              # (N, 1)
    ], axis=1).astype(np.float32)            # (N, 22)

    if config["subsample_ratio"] < 1.0:
        n_keep = int(len(states) * config["subsample_ratio"])
        indices = np.random.choice(len(states), size=n_keep, replace=False)
        indices.sort()
        states = states[indices]
        novice_actions = novice_actions[indices]
        expert_actions = expert_actions[indices]
        deltas = deltas[indices]
        rewards = rewards[indices]
        coach_inputs = coach_inputs[indices]

    print(f"  [OK]   {run_name} (seed={seed}): eval={success_rate:.2f}, "
          f"novice_eps={len(novice_episodes)}, samples={len(states)}")

    return {
        "states": states,
        "novice_actions": novice_actions,
        "expert_actions": expert_actions,
        "deltas": deltas,
        "rewards": rewards,
        "coach_inputs": coach_inputs,
    }


def main():
    parser = argparse.ArgumentParser(description="Build offline BC dataset for Coach V1")
    parser.add_argument("--logs_dir", type=str, default=None,
                        help="Path to logs_student directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to save dataset")
    parser.add_argument("--min_eval_success", type=float, default=0.80,
                        help="Minimum eval success rate to qualify as expert")
    parser.add_argument("--novice_ratio", type=float, default=0.25,
                        help="First N%% of episodes treated as novice (0.0-1.0)")
    parser.add_argument("--skip_initial_steps", type=int, default=5,
                        help="Skip first K steps per episode (reset transients)")
    parser.add_argument("--subsample_ratio", type=float, default=1.0,
                        help="Subsample ratio for dataset size control")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only show which runs would be processed, don't build dataset")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[3]

    if args.logs_dir is None:
        logs_dir = project_root / "train" / "unicycle_train" / "student" / "logs_student"
    else:
        logs_dir = Path(args.logs_dir)

    if args.output_dir is None:
        output_dir = script_dir / "datasets"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    config = DEFAULT_CONFIG.copy()
    config["min_eval_success_rate"] = args.min_eval_success
    config["novice_ratio"] = args.novice_ratio
    config["skip_initial_steps"] = args.skip_initial_steps
    config["subsample_ratio"] = args.subsample_ratio

    np.random.seed(args.seed)

    print("=" * 70)
    print("Coach V1 - Offline BC Dataset Builder")
    print("=" * 70)
    print(f"  Logs dir:           {logs_dir}")
    print(f"  Output dir:         {output_dir}")
    print(f"  Min eval success:   {config['min_eval_success_rate']:.2f}")
    print(f"  Novice ratio:       {config['novice_ratio']:.2f}")
    print(f"  Skip initial steps: {config['skip_initial_steps']}")
    print(f"  Subsample ratio:    {config['subsample_ratio']:.2f}")
    print(f"  Coach input dim:    22 (state 19 + action 2 + reward 1)")
    print(f"  Seed:               {args.seed}")
    print(f"  Dry run:            {args.dry_run}")
    print("=" * 70)

    run_dirs = sorted([
        d for d in logs_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ])
    print(f"\nFound {len(run_dirs)} run directories")

    if args.dry_run:
        print("\n[DRY RUN] Checking which runs qualify...\n")
        for run_dir in run_dirs:
            eval_results = load_eval_results(run_dir)
            seed = load_seed(run_dir)
            if eval_results:
                sr = eval_results.get("current_success_rate", 0.0)
                status = "PASS" if sr >= config["min_eval_success_rate"] else "SKIP"
                print(f"  [{status}] {run_dir.name} (seed={seed}): eval={sr:.2f}")
            else:
                print(f"  [SKIP] {run_dir.name}: no eval results")
        return

    print("\nProcessing students...\n")
    all_data = {
        "states": [],
        "novice_actions": [],
        "expert_actions": [],
        "deltas": [],
        "rewards": [],
        "coach_inputs": [],
    }
    student_stats = []
    start_time = time.time()

    for run_dir in run_dirs:
        result = process_single_student(run_dir, config)
        if result is not None:
            n_samples = len(result["states"])
            for key in all_data:
                all_data[key].append(result[key])
            student_stats.append({
                "run": run_dir.name,
                "seed": load_seed(run_dir),
                "eval_success": load_eval_results(run_dir).get("current_success_rate", 0.0),
                "n_samples": n_samples,
            })

    if not student_stats:
        print("\n[ERROR] No valid students found. Check --min_eval_success or data.")
        return

    dataset = {
        key: np.concatenate(all_data[key], axis=0)
        for key in all_data
    }

    elapsed = time.time() - start_time
    n_total = len(dataset["states"])
    n_students = len(student_stats)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = f"bc_dataset_{timestamp}"
    dataset_path = output_dir / f"{dataset_name}.npz"
    config_path = output_dir / f"{dataset_name}_config.json"
    stats_path = output_dir / f"{dataset_name}_stats.json"

    np.savez_compressed(
        dataset_path,
        states=dataset["states"],
        novice_actions=dataset["novice_actions"],
        expert_actions=dataset["expert_actions"],
        deltas=dataset["deltas"],
        rewards=dataset["rewards"],
        coach_inputs=dataset["coach_inputs"],
    )

    save_config = {
        **config,
        "coach_input_dim": 22,
        "coach_input_format": "state(19) + student_action(2) + reward(1)",
        "build_timestamp": timestamp,
        "build_seed": args.seed,
        "logs_dir": str(logs_dir),
        "n_students": n_students,
        "n_total_samples": n_total,
        "build_time_seconds": round(elapsed, 2),
    }
    with open(config_path, 'w') as f:
        json.dump(save_config, f, indent=2)

    with open(stats_path, 'w') as f:
        json.dump(student_stats, f, indent=2)

    print("\n" + "=" * 70)
    print("DATASET BUILD COMPLETE")
    print("=" * 70)
    print(f"  Students used:    {n_students} / {len(run_dirs)}")
    print(f"  Total samples:    {n_total:,}")
    print(f"  Coach input dim:  {dataset['coach_inputs'].shape[1]}")
    print(f"  Delta dim:        {dataset['deltas'].shape[1]}")
    print(f"  Build time:       {elapsed:.1f}s")
    print()
    print(f"  Dataset file:     {dataset_path}")
    print(f"  Config file:      {config_path}")
    print(f"  Stats file:       {stats_path}")
    print()

    deltas = dataset["deltas"]
    print("  Delta statistics:")
    print(f"    mean:  [{deltas[:, 0].mean():+.4f}, {deltas[:, 1].mean():+.4f}]")
    print(f"    std:   [{deltas[:, 0].std():.4f}, {deltas[:, 1].std():.4f}]")
    print(f"    |delta| mean: {np.linalg.norm(deltas, axis=1).mean():.4f}")
    print(f"    |delta| max:  {np.linalg.norm(deltas, axis=1).max():.4f}")
    print()

    rewards = dataset["rewards"]
    print("  Reward statistics:")
    print(f"    mean: {rewards.mean():+.4f}")
    print(f"    std:  {rewards.std():.4f}")
    print(f"    range: [{rewards.min():.4f}, {rewards.max():.4f}]")
    print()

    print("  Per-student breakdown:")
    for s in student_stats:
        print(f"    {s['run'][-4:]} (seed={s['seed']}): "
              f"eval={s['eval_success']:.2f}, samples={s['n_samples']:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()