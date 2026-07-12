# train/unicycle_train/diagnostics/check_demo_inject.py
#
# Smoke-test checker for demo injection runs.
# For each run directory it verifies:
#   1. Demo episodes occur exactly on schedule (prefix or every-N pattern)
#   2. Demo episode success rate (~100% expected -> wrapper plumbing is correct)
#   3. Budget accounting: total / demo / student episode counts
#   4. Student-episode stats (length, success) -- informational only, NOT for judgment
#
# Usage:
#   python -m train.unicycle_train.diagnostics.check_demo_inject \
#       train/unicycle_train/diagnostics/data/student_demo_inject/seed42_*

import argparse
import re
from pathlib import Path

import pandas as pd


def load_monitor(run_dir: Path) -> pd.DataFrame:
    csvs = sorted(run_dir.glob("monitor*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No monitor csv in {run_dir}")
    # SB3 Monitor csv has a '#{json}' header line -> skiprows=1
    df = pd.read_csv(csvs[0], skiprows=1)
    df["ep_idx"] = range(len(df))
    # Monitor may serialize booleans as strings
    for col in ("coach_episode", "goal_reached", "is_fallen"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin(("true", "1", "1.0"))
    return df


def infer_expected_schedule(run_name: str, n_episodes: int):
    """Reconstruct expected demo episode indices from the run directory name."""
    m = re.search(r"demoevery(\d+)", run_name)
    if m:
        k = int(m.group(1))
        return {i for i in range(n_episodes) if i % k == 0}, f"demo_every={k}"
    m = re.search(r"prefix(\d+)", run_name)
    if m:
        n = int(m.group(1))
        return set(range(n)), f"prefix_demos={n}"
    return None, "unknown"


def check_run(run_dir: Path):
    df = load_monitor(run_dir)
    n = len(df)
    print(f"\n{'=' * 70}")
    print(f"RUN: {run_dir.name}   ({n} episodes logged)")
    print(f"{'=' * 70}")

    if "coach_episode" not in df.columns:
        print("  [FAIL] coach_episode column missing from monitor csv")
        return

    demo = df[df["coach_episode"]]
    student = df[~df["coach_episode"]]

    # --- 1. schedule verification ---
    expected, label = infer_expected_schedule(run_dir.name, n)
    actual = set(demo["ep_idx"])
    if expected is not None:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        ok = not missing and not extra
        print(f"  [{'OK' if ok else 'FAIL'}] schedule ({label}): "
              f"{len(actual)} demos, expected {len(expected)}")
        if missing:
            print(f"         missing demo episodes: {missing[:10]}")
        if extra:
            print(f"         unexpected demo episodes: {extra[:10]}")
    else:
        print(f"  [??] could not infer schedule from name; demos at {sorted(actual)[:10]}...")

    # --- 2. demo success rate (the critical sanity check) ---
    if "goal_reached" in df.columns and len(demo):
        rate = demo["goal_reached"].mean()
        flag = "OK" if rate >= 0.95 else "FAIL"
        print(f"  [{flag}] demo success rate: {rate:.3f} "
              f"({int(demo['goal_reached'].sum())}/{len(demo)})")
        if rate < 0.95:
            print("         -> wrapper obs plumbing is suspect (last_obs timing?)")

    # --- 3. budget accounting ---
    print(f"  [--] budget: total={n}  demo={len(demo)}  student={len(student)}")

    # --- 4. student-episode stats (informational only) ---
    if len(student):
        third = max(1, len(student) // 3)
        for name, chunk in [("early", student.iloc[:third]),
                            ("mid", student.iloc[third:2 * third]),
                            ("late", student.iloc[2 * third:])]:
            succ = chunk["goal_reached"].mean() if "goal_reached" in chunk else float("nan")
            print(f"  [--] student {name:5s}: mean_len={chunk['l'].mean():7.1f}  "
                  f"success={succ:.3f}  (n={len(chunk)})")
        print("       (informational only -- single seed, not for judgment)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    args = ap.parse_args()
    for d in args.run_dirs:
        check_run(d)


if __name__ == "__main__":
    main()