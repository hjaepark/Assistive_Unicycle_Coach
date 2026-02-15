# train/unicycle_train/student/data/label_rl_students.py

import os
import glob
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

from train.unicycle_train.path import LOGS_STUDENT_DIR, DATA_STUDENT_DIR


DELTA = 0.10  # for dead-zone
LOGS_DIR = LOGS_STUDENT_DIR
OUT_DIR = DATA_STUDENT_DIR
OUT_CSV = OUT_DIR / "rl_student_trajectory_metrics.csv"
OUT_CONFIG = OUT_DIR / "label_config.yaml"

# Filter run folders
RUN_GLOB = "run_*20260213_*"


def read_seed(run_dir: Path) -> Optional[int]:
    seed_path = run_dir / "seed.txt"
    if not seed_path.exists():
        return None
    try:
        txt = seed_path.read_text().strip()
        return int(txt)
    except Exception:
        return None


def sgn_delta(u: np.ndarray, delta: float) -> np.ndarray:
    """dead-zone function: +1, -1, 0"""
    out = np.zeros_like(u, dtype=np.int8)
    out[u > delta] = 1
    out[u < -delta] = -1
    return out


def flip_rate_1d(a: np.ndarray, delta: float) -> float:

    b = sgn_delta(a.astype(np.float64), delta)
    # remove zeros
    nz = b[b != 0]
    k = len(nz)
    if k <= 1:
        return 0.0
    flips = np.sum(nz[1:] != nz[:-1])
    return float(flips) / float(max(1, k - 1))


def termination_label(last_row: pd.Series) -> str:
    # success > fall > timeout > other_done
    if bool(last_row.get("success_indicator", False)):
        return "success"
    if bool(last_row.get("is_fallen", False)):
        return "fall"
    if bool(last_row.get("truncated", False)):
        return "timeout"
    if bool(last_row.get("done", False)):
        return "other_done"
    return "incomplete"  # Drop episode


def write_label_config_yaml(path: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    content = f"""# Auto-generated label config
version: 1
delta: {DELTA}

episode_group_key: episode
drop_incomplete_episode_rule: "drop if last row has (done==False AND truncated==False)"

distance:
  column: obs_0
  metric_name: maximum_distance
  definition: "max(obs_0) - first(obs_0) within episode"

action:
  columns: [final_action_0, final_action_1]
  metric_names:
    flip_rate: action_flip_rate
    effort: control_effort

action_flip_rate:
  definition: "per-channel sgn_delta -> remove zeros -> sign-change rate; then average over 2 channels"
  sgn_delta:
    +1: "u > delta"
    -1: "u < -delta"
    0: "|u| <= delta"

control_effort:
  definition: "mean( a0^2 + a1^2 ) over episode"

termination_label:
  priority: [success_indicator, is_fallen, truncated, done]
  labels:
    success_indicator: success
    is_fallen: fall
    truncated: timeout
    done: other_done
"""
    path.write_text(content)

def process_run(run_dir: Path) -> pd.DataFrame:
    traj_path = run_dir / "trajectory.csv"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectory.csv in {run_dir}")

    df = pd.read_csv(traj_path)

    required_cols = [
        "participant_id", "condition", "phase",
        "episode", "obs_0",
        "final_action_0", "final_action_1",
        "done", "truncated", "success_indicator", "is_fallen",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{run_dir.name}: missing columns: {missing}")

    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df = df.dropna(subset=["episode"])
    df["episode"] = df["episode"].astype(int)

    run_id = run_dir.name
    seed = read_seed(run_dir)

    rows: List[Dict[str, Any]] = []


    group_keys = ["participant_id", "condition", "phase", "episode"]
    for (pid, cond, phase, ep), g in df.groupby(group_keys, sort=True):
        last = g.iloc[-1]
        done_last = bool(last.get("done", False))
        truncated_last = bool(last.get("truncated", False))
        if (not done_last) and (not truncated_last):
            continue  # drop incomplete episode

        # termination label
        term = termination_label(last)
        if term == "incomplete":
            continue

        # T: episode length (row count)
        T = int(len(g))

        # maximum_distance: max(obs_0) - first(obs_0)
        x0 = float(g["obs_0"].iloc[0])
        max_x = float(g["obs_0"].max())
        maximum_distance = max_x - x0

        # action arrays
        a0 = g["final_action_0"].to_numpy(dtype=np.float64)
        a1 = g["final_action_1"].to_numpy(dtype=np.float64)

        # action_flip_rate
        f0 = flip_rate_1d(a0, DELTA)
        f1 = flip_rate_1d(a1, DELTA)
        action_flip_rate = 0.5 * (f0 + f1)

        # control_effort = mean(a0^2 + a1^2)
        control_effort = float(np.mean(a0 * a0 + a1 * a1))

        rows.append({
            "run_id": run_id,
            "seed": seed,
            "participant_id": pid,
            "condition": cond,
            "phase": phase,
            "episode": ep,
            "T": T,
            "termination": term,
            "maximum_distance": maximum_distance,
            "action_flip_rate": action_flip_rate,
            "control_effort": control_effort,
        })

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([Path(p) for p in glob.glob(str(LOGS_DIR / RUN_GLOB)) if Path(p).is_dir()])
    if not run_dirs:
        raise RuntimeError(f"No run directories found under {LOGS_DIR} with pattern {RUN_GLOB}")

    all_dfs = []
    for rd in run_dirs:
        try:
            ep_df = process_run(rd)
            if len(ep_df) > 0:
                all_dfs.append(ep_df)
            print(f"[OK] {rd.name}: episodes={len(ep_df)}")
        except Exception as e:
            print(f"[SKIP/ERR] {rd.name}: {e}")

    if not all_dfs:
        raise RuntimeError("No episodes collected. Check filters / column names / completeness rule.")

    out_df = pd.concat(all_dfs, ignore_index=True)

    out_df = out_df.sort_values(["run_id", "episode"]).reset_index(drop=True)

    out_df.to_csv(OUT_CSV, index=False)
    write_label_config_yaml(OUT_CONFIG)

    print("\nSaved:")
    print(f" - {OUT_CSV}  (rows={len(out_df)})")
    print(f" - {OUT_CONFIG}")


if __name__ == "__main__":
    main()
