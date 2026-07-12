# train/unicycle_train/diagnostics/e0_criterion_analysis.py
#
# E0: can early-training behavior predict the final solo outcome?
#
# Uses ONLY existing baseline data (monitor logs + eval.npz). For each
# candidate metric m and each checkpoint episode t, computes the rolling
# value m_t per seed, then measures how well m_t separates the final
# STRONG group (eval >= 0.5) from the final WEAK group (eval == 0).
#
# Separation is reported as AUC (probability that a random strong seed
# scores higher than a random weak seed; 0.5 = useless, 1.0 = perfect).
# For informative rows a suggested threshold (theta) is printed: the
# midpoint between the weak-group 90th percentile and strong-group 10th
# percentile when they don't overlap, else the best-accuracy split.
#
# Usage:
#   python -m train.unicycle_train.diagnostics.e0_criterion_analysis \
#       train/unicycle_train/diagnostics/data/student_baseline

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

W = 20                                # rolling window (episodes)
CHECKPOINTS = [200, 250, 300, 350, 400, 450]


def runs(exp_dir: Path):
    latest = {}
    for d in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        m = re.search(r"seed(\d+)", d.name)
        if m:
            latest[int(m.group(1))] = d
    return latest


def load(run_dir: Path):
    csv = sorted(run_dir.glob("monitor*.csv"))
    df = pd.read_csv(csv[0], skiprows=1)
    if df["goal_reached"].dtype == object:
        df["goal_reached"] = df["goal_reached"].astype(str).str.lower().isin(
            ("true", "1", "1.0"))
    return df


def metric_at(df, t, name):
    """Rolling metric over episodes [t-W, t)."""
    win = df.iloc[max(0, t - W):t]
    if len(win) < W:
        return None
    if name == "success":
        return win["goal_reached"].mean()
    if name == "length":
        return win["l"].mean()
    if name == "slope":
        # linear trend of episode length inside the window
        return np.polyfit(np.arange(len(win)), win["l"].to_numpy(), 1)[0]
    raise ValueError(name)


def auc(strong, weak):
    """Mann-Whitney AUC: P(strong sample > weak sample)."""
    s, w = np.asarray(strong), np.asarray(weak)
    wins = (s[:, None] > w[None, :]).sum() + 0.5 * (s[:, None] == w[None, :]).sum()
    return wins / (len(s) * len(w))


def suggest_theta(strong, weak):
    s, w = np.sort(strong), np.sort(weak)
    lo, hi = np.percentile(w, 90), np.percentile(s, 10)
    if hi > lo:
        return 0.5 * (lo + hi)
    # overlapping: pick split maximizing classification accuracy
    cands = np.unique(np.concatenate([s, w]))
    accs = [((s >= c).mean() + (w < c).mean()) / 2 for c in cands]
    return cands[int(np.argmax(accs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir", type=Path)
    args = ap.parse_args()

    dirs = runs(args.baseline_dir)
    final = {s: float(np.load(d / "eval.npz")["successes"].mean())
             for s, d in dirs.items() if (d / "eval.npz").exists()}
    strong = [s for s, v in final.items() if v >= 0.5]
    weak = [s for s, v in final.items() if v == 0.0]
    print(f"strong (eval>=0.5): {len(strong)} seeds   "
          f"weak (eval==0): {len(weak)} seeds   "
          f"(middle excluded: {len(final) - len(strong) - len(weak)})\n")

    mons = {s: load(d) for s, d in dirs.items() if s in strong + weak}

    header = f"{'ep':>4}  {'metric':<8}  {'strong mean':>11}  {'weak mean':>9}  {'AUC':>5}  {'theta*':>8}"
    print(header)
    print("-" * len(header))
    for t in CHECKPOINTS:
        for name in ("success", "length", "slope"):
            sv = [metric_at(mons[s], t, name) for s in strong]
            wv = [metric_at(mons[s], t, name) for s in weak]
            sv = [v for v in sv if v is not None]
            wv = [v for v in wv if v is not None]
            if not sv or not wv:
                continue
            a = auc(sv, wv)
            th = suggest_theta(sv, wv) if a >= 0.75 else None
            print(f"{t:>4}  {name:<8}  {np.mean(sv):>11.2f}  {np.mean(wv):>9.2f}  "
                  f"{a:>5.2f}  {th if th is None else f'{th:>8.2f}'}")
        print()


if __name__ == "__main__":
    main()