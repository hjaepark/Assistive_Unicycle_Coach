# train/estimator/features.py
"""Data loading layer for the skill estimator.

Primary path: read per-episode distance straight out of the Recorder npz dumps.
`OBS_SCHEMA[0] == "pos_x"`, so distance is recorded exactly and needs no
reconstruction.

Fallback path: recover distance by inverting the reward from monitor.monitor.csv.
Needed because `*.npz` is gitignored -- anyone who clones the repo, or who has
pruned old episode dumps, can still reproduce the validation from the committed
monitor logs. Accurate to <1mm but it hardcodes the reward constants, so it
breaks silently if REWARD changes.
"""

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

BASELINE_DIR = Path(__file__).resolve().parents[1] / "student" / "data" / "baseline"

POS_X = 0          # index of pos_x in OBS_SCHEMA
GOAL_X = 5.0       # course length

# --- reward constants, only used by the fallback path ---
PROGRESS_WEIGHT = 60.0
PROGRESS_EXP_K = 0.05
GOAL_BONUS = 50.0
FALL_PENALTY = -20.0
ASSUMED_ZERO_WEIGHTS = ("posture", "pendulum", "speed_reward_weight",
                        "alive_bonus", "upright_gate_k")


class RunData(NamedTuple):
    dist: np.ndarray       # net displacement per episode (m)
    dist_max: np.ndarray   # furthest point reached per episode (m)
    goal: np.ndarray       # 1.0 if the finish line was crossed
    fall: np.ndarray       # 1.0 if the student fell
    source: str            # "npz" or "monitor"


# ---------------------------------------------------------------- npz path

def episode_features(npz_path):
    """Exact distance and outcome flags for one recorded episode."""
    z = np.load(npz_path)
    x = z["observations"][:, POS_X]
    return (float(x[-1] - x[0]),
            float(x.max() - x[0]),
            float(bool(z["goal_reached"])),
            float(bool(z["is_fallen"])))


def load_run_npz(run_dir, n_episodes=None):
    """Read a run's episodes/ directory. Returns RunData or None if absent."""
    ep_dir = Path(run_dir) / "episodes"
    files = sorted(ep_dir.glob("ep*.npz")) if ep_dir.is_dir() else []
    if not files:
        return None
    if n_episodes is not None:
        files = files[:n_episodes]
    rows = [episode_features(f) for f in files]
    d, dmax, g, f = (np.array(c, dtype=float) for c in zip(*rows))
    return RunData(d, dmax, g, f, "npz")


# ------------------------------------------------------------ monitor path

def verify_reward_config(reward_dict):
    """Raise if the reward config is incompatible with recover_distance()."""
    bad = [k for k in ASSUMED_ZERO_WEIGHTS if reward_dict.get(k, 0.0) != 0.0]
    if bad:
        raise ValueError(
            f"recover_distance() assumes these weights are 0 but they are not: {bad}. "
            "Use the npz path instead."
        )
    for key, expected in (("progress", PROGRESS_WEIGHT),
                          ("goal_bonus", GOAL_BONUS),
                          ("fall_penalty", FALL_PENALTY)):
        if reward_dict.get(key) != expected:
            raise ValueError(
                f"reward['{key}']={reward_dict.get(key)} but features.py expects "
                f"{expected}. Update the constants in this module."
            )


def recover_distance(reward, goal_reached, is_fallen):
    """Invert `return = 60*(exp(0.05*dx)-1) + 50*goal - 20*fall` for dx."""
    reward = np.asarray(reward, dtype=float)
    goal = np.asarray(goal_reached, dtype=float)
    fall = np.asarray(is_fallen, dtype=float)
    progress = reward - GOAL_BONUS * goal - FALL_PENALTY * fall
    ratio = np.clip(1.0 + progress / PROGRESS_WEIGHT, 1e-6, None)
    return np.log(ratio) / PROGRESS_EXP_K


def _as_bool(series):
    return series.astype(str).str.strip().isin(["True", "1", "1.0"]).values.astype(float)


def load_run_monitor(run_dir, n_episodes=None):
    """Reconstruct RunData from monitor.monitor.csv. Returns None if absent."""
    path = Path(run_dir) / "monitor.monitor.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, skiprows=1)          # first line is a JSON header
    if n_episodes is not None:
        df = df.iloc[:n_episodes]
    goal, fall = _as_bool(df["goal_reached"]), _as_bool(df["is_fallen"])
    d = recover_distance(df["r"].values, goal, fall)
    # net displacement is all the reward encodes; max is unavailable here
    return RunData(d, d.copy(), goal, fall, "monitor")


# ------------------------------------------------------------------ pool

def load_run(run_dir, n_episodes=None, prefer="npz"):
    """Load one run, preferring npz and falling back to the monitor log."""
    order = (load_run_npz, load_run_monitor) if prefer == "npz" else \
            (load_run_monitor, load_run_npz)
    for fn in order:
        out = fn(run_dir, n_episodes)
        if out is not None:
            return out
    return None


def load_pool(root=None, n_episodes=300, prefer="npz", cache=True, verbose=True):
    """Load every run under `root` into {seed: RunData}.

    Reading 100 seeds x 300 npz files is slow, so the result is cached to
    `<root>/../estimator_pool_cache.npz`. Delete that file to force a reread.
    """
    root = Path(root or BASELINE_DIR)
    cache_path = root.parent / f"estimator_pool_cache_{prefer}_{n_episodes}.npz"

    if cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        seeds = z["seeds"]
        pool = {int(s): RunData(z["dist"][i], z["dist_max"][i], z["goal"][i],
                                z["fall"][i], str(z["source"].item()))
                for i, s in enumerate(seeds)}
        if verbose:
            print(f"[cache] {len(pool)} seeds from {cache_path.name}")
        return pool

    pool = {}
    for d in sorted(root.glob("seed*")):
        m = re.search(r"seed(\d+)", d.name)
        if not m:
            continue
        run = load_run(d, n_episodes, prefer)
        if run is None or len(run.dist) < n_episodes:
            continue
        pool[int(m.group(1))] = run          # later dirs win for duplicate seeds

    if not pool:
        return pool
    if verbose:
        src = next(iter(pool.values())).source
        print(f"[load] {len(pool)} seeds from {root} (source={src})")

    if cache:
        seeds = sorted(pool)
        np.savez_compressed(
            cache_path, seeds=np.array(seeds),
            dist=np.array([pool[s].dist for s in seeds]),
            dist_max=np.array([pool[s].dist_max for s in seeds]),
            goal=np.array([pool[s].goal for s in seeds]),
            fall=np.array([pool[s].fall for s in seeds]),
            source=np.array(next(iter(pool.values())).source))
        if verbose:
            print(f"[cache] wrote {cache_path.name}")
    return pool


def stack(pool, field="dist"):
    """{seed: RunData} -> (seeds, matrix) with rows ordered by seed."""
    seeds = sorted(pool)
    return seeds, np.array([getattr(pool[s], field) for s in seeds])