# data/label_human_student.py

import glob
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

DELTA = 0.10  # dead-zone
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMAN_DATA_DIR = PROJECT_ROOT / "data"

PHASE_NAME = "training"
P_GLOB = "P??"

ACTION0 = "human_action_0"
ACTION1 = "human_action_1"

# forward position
XCOL = "obs_0"


def sgn_delta(u: np.ndarray, delta: float) -> np.ndarray:
    out = np.zeros_like(u, dtype=np.int8)
    out[u > delta] = 1
    out[u < -delta] = -1
    return out


def flip_rate_1d(a: np.ndarray, delta: float) -> float:
    b = sgn_delta(a.astype(np.float64), delta)
    nz = b[b != 0]
    k = len(nz)
    if k <= 1:
        return 0.0
    flips = np.sum(nz[1:] != nz[:-1])
    return float(flips) / float(max(1, k - 1))


def termination_label(last_row: pd.Series) -> str:
    if bool(last_row.get("success_indicator", False)):
        return "success"
    if bool(last_row.get("is_fallen", False)):
        return "fall"
    if bool(last_row.get("truncated", False)):
        return "timeout"
    if bool(last_row.get("done", False)):
        return "other_done"
    return "incomplete"


def process_one_participant(p_dir: Path) -> pd.DataFrame:
    pid = p_dir.name  # "P01"
    traj_path = p_dir / f"{pid}_{PHASE_NAME}_trajectory.csv"
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)

    df = pd.read_csv(traj_path)

    required_cols = [
        "participant_id", "condition", "phase",
        "episode", XCOL,
        ACTION0, ACTION1,
        "done", "truncated", "success_indicator", "is_fallen",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{pid}: missing columns: {missing}")

    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df = df.dropna(subset=["episode"])
    df["episode"] = df["episode"].astype(int)

    rows: List[Dict[str, Any]] = []

    group_keys = ["participant_id", "condition", "phase", "episode"]
    for (pid_infile, cond, phase, ep), g in df.groupby(group_keys, sort=True):
        last = g.iloc[-1]
        done_last = bool(last.get("done", False))
        truncated_last = bool(last.get("truncated", False))
        if (not done_last) and (not truncated_last):
            continue  # drop incomplete episode

        term = termination_label(last)
        if term == "incomplete":
            continue

        T = int(len(g))

        x0 = float(g[XCOL].iloc[0])
        max_x = float(g[XCOL].max())
        maximum_distance = max_x - x0

        a0 = g[ACTION0].to_numpy(dtype=np.float64)
        a1 = g[ACTION1].to_numpy(dtype=np.float64)

        f0 = flip_rate_1d(a0, DELTA)
        f1 = flip_rate_1d(a1, DELTA)
        action_flip_rate = 0.5 * (f0 + f1)

        control_effort = float(np.mean(a0 * a0 + a1 * a1))

        rows.append({
            "run_id": pid,               
            "seed": None,                
            "participant_id": pid_infile,
            "condition": cond,
            "phase": phase,
            "episode": ep,
            "T": T,
            "termination": term,
            "maximum_distance": maximum_distance,
            "action_flip_rate": action_flip_rate,
            "control_effort": control_effort,
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values(["run_id", "episode"]).reset_index(drop=True)
    return out


def main() -> None:
    p_dirs = sorted([p for p in HUMAN_DATA_DIR.glob(P_GLOB) if p.is_dir()])
    if not p_dirs:
        raise RuntimeError(f"No participant dirs found under {HUMAN_DATA_DIR}")

    all_dfs = []
    for p_dir in p_dirs:
        pid = p_dir.name
        try:
            ep_df = process_one_participant(p_dir)
            out_path = p_dir / f"{pid}_{PHASE_NAME}_trajectory_metrics.csv"
            ep_df.to_csv(out_path, index=False)
            print(f"[OK] {pid}: episodes={len(ep_df)} -> {out_path.name}")
            all_dfs.append(ep_df)
        except Exception as e:
            print(f"[SKIP/ERR] {pid}: {e}")

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged_out = HUMAN_DATA_DIR / f"human_{PHASE_NAME}_trajectory_metrics.csv"
        merged.to_csv(merged_out, index=False)
        print(f"\nSaved merged: {merged_out} (rows={len(merged)})")


if __name__ == "__main__":
    main()