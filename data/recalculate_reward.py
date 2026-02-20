# data/recalculate_reward.py

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# New reward parameters
PROGRESS_WEIGHT = 60.0
PROGRESS_EXP_K = 0.05
GOAL_BONUS = 200.0
FALL_PENALTY = -20.0
FINISH_LINE_X = 5.0

# upright_gate_k=0 → upright=exp(0)=1.0 always, so we skip that computation
# posture=0, pendulum=0, speed=0, alive_bonus=0 → all zero


def recalculate_episode_rewards(ep_df: pd.DataFrame) -> np.ndarray:
    x_positions = ep_df["obs_0"].values
    is_fallen_flags = ep_df["is_fallen"].values
    done_flags = ep_df["done"].values

    n = len(ep_df)
    rewards = np.zeros(n, dtype=np.float64)

    # start_x_pos = first timestep's x position (matches env reset behavior)
    start_x_pos = x_positions[0]
    prev_x_pos = start_x_pos

    # Track if fall already happened (env only counts first fall)
    fallen_triggered = False

    for i in range(n):
        x_pos = x_positions[i]
        dx = x_pos - prev_x_pos

        # Progress reward (upright gate = 1.0 since upright_gate_k = 0)
        k = PROGRESS_EXP_K
        base_arg = np.clip(k * (prev_x_pos - start_x_pos), -60.0, 60.0)
        growth_arg = np.clip(k * dx, -10.0, 10.0)
        exp_progress = np.exp(base_arg) * np.expm1(growth_arg)
        progress_reward = PROGRESS_WEIGHT * float(exp_progress)
        # upright=1.0 so no gating needed

        # Goal bonus
        crossed = (prev_x_pos < FINISH_LINE_X <= x_pos)
        goal_bonus = GOAL_BONUS if crossed else 0.0

        # Fall penalty
        current_fallen = bool(is_fallen_flags[i])
        fall_penalty = 0.0
        if current_fallen and not fallen_triggered:
            fall_penalty = FALL_PENALTY
            fallen_triggered = True

        # Total reward
        rewards[i] = progress_reward + goal_bonus + fall_penalty

        prev_x_pos = x_pos

    return rewards


def process_participant(p_dir: Path) -> bool:
    pid = p_dir.name  # e.g. "P01"
    input_csv = p_dir / f"{pid}_training_trajectory.csv"

    if not input_csv.exists():
        print(f"  [skip] {input_csv} not found")
        return False

    df = pd.read_csv(input_csv)
    original_reward_sum = df["reward"].sum()

    # Process episode by episode
    episodes = df["episode"].unique()
    print(f"  {pid}: {len(df)} rows, {len(episodes)} episodes")

    new_rewards = np.zeros(len(df), dtype=np.float64)

    for ep_id in episodes:
        ep_mask = df["episode"] == ep_id
        ep_df = df[ep_mask]
        ep_rewards = recalculate_episode_rewards(ep_df)
        new_rewards[ep_mask.values] = ep_rewards

    # Replace reward column
    df["reward"] = new_rewards

    # Save updated file (original untouched)
    output_csv = p_dir / f"{pid}_training_trajectory_updated.csv"
    df.to_csv(output_csv, index=False)

    new_reward_sum = df["reward"].sum()
    print(f"  → Saved: {output_csv.name}")
    print(f"    reward sum: {original_reward_sum:.2f} → {new_reward_sum:.2f}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Recalculate rewards with max-success-rate function")
    parser.add_argument("--data_dir", type=str, default=".",
                        help="Path to data directory containing P01-P10 folders")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[info] Data directory: {data_dir.resolve()}")
    print(f"[info] New reward weights: progress={PROGRESS_WEIGHT}, "
          f"goal_bonus={GOAL_BONUS}, fall_penalty={FALL_PENALTY}")
    print(f"[info] All other weights = 0 (posture, pendulum, speed, alive_bonus)\n")

    success_count = 0
    for i in range(1, 11):
        pid = f"P{i:02d}"
        p_dir = data_dir / pid
        if p_dir.is_dir():
            if process_participant(p_dir):
                success_count += 1
        else:
            print(f"  [skip] {p_dir} directory not found")

    print(f"\n[done] Processed {success_count}/10 participants")
    print(f"[note] Original files untouched. Updated files: *_training_trajectory_updated.csv")


if __name__ == "__main__":
    main()