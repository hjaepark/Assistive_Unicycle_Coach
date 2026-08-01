# train/estimator/evaluate.py
"""Validation harness for the skill estimator.

Answers, with seed-level cross-validation so nothing about a held-out seed's
future touches the fit:

  1. Can we discriminate students who will not make it alone?   -> AUC
  2. Are the reported probabilities honest?                     -> Brier + reliability
  3. How few episodes do we need?                               -> window sweep
  4. Is net displacement or furthest-point-reached better?      -> field comparison

Run: python -m train.estimator.evaluate
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import KFold

from .estimator import POOR_THRESHOLD, TARGET_WINDOW, build_targets
from .features import BASELINE_DIR, load_pool, stack

WINDOWS = (10, 20, 30, 50, 75, 100)
N_SPLITS = 5
SEED = 0


def oof_probabilities(x, poor):
    """Out-of-fold P(poor) from a single scalar feature."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    p = np.zeros(len(x))
    for tr, te in KFold(N_SPLITS, shuffle=True, random_state=SEED).split(x):
        lr = LogisticRegression().fit(x[tr], poor[tr])
        p[te] = lr.predict_proba(x[te])[:, 1]
    return p


def reliability(p, poor, bins=((0, .2), (.2, .5), (.5, .8), (.8, 1.01))):
    rows = []
    for lo, hi in bins:
        m = (p >= lo) & (p < hi)
        if m.sum() >= 4:
            rows.append((lo, hi, int(m.sum()), float(p[m].mean()), float(poor[m].mean())))
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", default=None)
    ap.add_argument("--prefer", default="npz", choices=["npz", "monitor"])
    args = ap.parse_args()

    pool = load_pool(args.baseline_dir or BASELINE_DIR, prefer=args.prefer)
    if not pool:
        raise SystemExit("no runs found")

    seeds, D = stack(pool, "dist")
    _, Dmax = stack(pool, "dist_max")
    _, G = stack(pool, "goal")
    y = build_targets(pool)
    poor = (y < POOR_THRESHOLD).astype(int)

    print(f"{len(seeds)} seeds, {D.shape[1]} episodes, {poor.sum()} poor students")
    print(f"target = sustained success over episodes "
          f"{TARGET_WINDOW[0]}-{TARGET_WINDOW[1]}, poor if < {POOR_THRESHOLD}\n")

    # --- 1, 3, 4: discrimination vs observation budget ---
    print("=== AUC for detecting students who will not make it alone ===")
    cols = [("success rate (current)", lambda W: G[:, :W].mean(1)),
            ("net distance", lambda W: D[:, :W].mean(1)),
            ("max distance", lambda W: Dmax[:, :W].mean(1))]
    print(f"{'W':>5}" + "".join(f"{n:>24s}" for n, _ in cols) + f"{'rho(net)':>10}")
    print("-" * 88)
    for W in WINDOWS:
        row = f"{W:>5}"
        for _, fn in cols:
            row += f"{roc_auc_score(poor, oof_probabilities(fn(W), poor)):>24.3f}"
        row += f"{stats.spearmanr(D[:, :W].mean(1), y)[0]:>10.3f}"
        print(row)

    # --- 2: are the probabilities honest? ---
    print("\n=== Calibration (net distance) ===")
    for W in (10, 30, 50):
        p = oof_probabilities(D[:, :W].mean(1), poor)
        print(f"  W={W:<4} Brier={brier_score_loss(poor, p):.3f}")
        for lo, hi, n, pred, obs in reliability(p, poor):
            flag = "ok" if abs(pred - obs) < 0.15 else "OFF"
            print(f"    predicted {lo:.1f}-{hi:.1f} (n={n:<3}) "
                  f"mean={pred:.2f}  actual={obs:.2f}  [{flag}]")

    # --- reference: how predictable is this at all? ---
    print("\n=== Intrinsic ceiling (reference, not an estimator) ===")
    prev, late = G[:, 200:250].mean(1), G[:, 250:300].mean(1)
    print(f"  adjacent 50-ep windows correlate rho={stats.spearmanr(prev, late)[0]:.3f}")
    print("  -> even perfect knowledge of current skill caps out near ~0.70")


if __name__ == "__main__":
    main()