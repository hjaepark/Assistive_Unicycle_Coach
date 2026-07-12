# train/unicycle_train/diagnostics/check_tail.py
#
# Inspects the tail of training: the last N student-controlled episodes.
# Purpose: detect end-of-training collapse that third-averaging hides.
# A healthy run ends with success/length comparable to its late-third stats;
# a collapsed run shows a cliff in the final few dozen episodes.
#
# Usage:
#   python -m train.unicycle_train.diagnostics.check_tail \
#       data/student_demo_inject/seed33_* data/student_demo_inject/seed97_* \
#       --tail 40

import argparse
from pathlib import Path

import pandas as pd


def load_monitor(run_dir: Path) -> pd.DataFrame:
    csvs = sorted(run_dir.glob("monitor*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No monitor csv in {run_dir}")
    df = pd.read_csv(csvs[0], skiprows=1)
    df["ep_idx"] = range(len(df))
    for col in ("coach_episode", "goal_reached", "is_fallen"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin(("true", "1", "1.0"))
    return df


def check_tail(run_dir: Path, tail: int):
    df = load_monitor(run_dir)
    student = df[~df["coach_episode"]] if "coach_episode" in df.columns else df

    print(f"\n{'=' * 70}")
    print(f"RUN: {run_dir.name}  (student episodes: {len(student)})")
    print(f"{'=' * 70}")

    # chunked view of the last `tail` episodes, 10 per row
    t = student.tail(tail)
    for start in range(0, len(t), 10):
        chunk = t.iloc[start:start + 10]
        marks = "".join("O" if g else "." for g in chunk["goal_reached"])
        print(f"  eps {chunk['ep_idx'].iloc[0]:3d}-{chunk['ep_idx'].iloc[-1]:3d}: "
              f"[{marks:<10s}]  success={chunk['goal_reached'].mean():.2f}  "
              f"mean_len={chunk['l'].mean():6.1f}")

    # summary: last-tail vs the late third excluding the tail
    late_third = student.iloc[2 * len(student) // 3:]
    body = late_third.iloc[:-tail] if len(late_third) > tail else late_third
    print(f"  ---")
    print(f"  late-third w/o tail: success={body['goal_reached'].mean():.3f}  "
          f"mean_len={body['l'].mean():6.1f}  (n={len(body)})")
    print(f"  final {tail:2d} episodes : success={t['goal_reached'].mean():.3f}  "
          f"mean_len={t['l'].mean():6.1f}")
    drop = body["goal_reached"].mean() - t["goal_reached"].mean()
    if drop > 0.2:
        print(f"  [!!] tail collapse detected (success dropped by {drop:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--tail", type=int, default=40,
                    help="number of final student episodes to inspect")
    args = ap.parse_args()
    for d in args.run_dirs:
        check_tail(d, args.tail)


if __name__ == "__main__":
    main()