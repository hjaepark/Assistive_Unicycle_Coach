# train/estimator/estimator.py
"""Skill estimator: mean travelled distance -> P(student will not succeed alone).

Design note. A local-linear-trend Kalman filter was implemented and rejected:
projecting the trend over the remaining ~250 episodes produced a term with
std 6.5m against a true skill spread of 1.9m, so extrapolation noise swamped
the signal (Spearman 0.27 vs 0.51 for a plain mean). A level-only random-walk
filter also lost, because student skill drifts slowly relative to the episode
noise, so discounting older episodes throws away information. Its uncertainty
was badly miscalibrated too (nominal 80% interval covered 59%).

What survived is a one-feature logistic model. It is calibrated, it beats the
current success-rate trigger in the early-decision regime that actually matters
for the coach, and it refits in milliseconds.
"""

import json
from pathlib import Path

import numpy as np

from .features import BASELINE_DIR, load_pool, stack

PARAMS_PATH = Path(__file__).resolve().parent / "params.json"

# A student is "poor" if their sustained late-training success rate is below
# this. Sustained = averaged over a wide window, because a narrow window is
# mostly measurement noise (adjacent 50-episode windows only correlate 0.66).
POOR_THRESHOLD = 0.33
TARGET_WINDOW = (200, 300)
DISTANCE_FIELD = "dist"          # "dist" (net) or "dist_max" (furthest reached)


class SkillEstimator:
    """Logistic model on mean travelled distance.

    Coefficients are stored per observation-window W, because calibration
    depends on how many episodes were averaged.
    """

    def __init__(self, coefs=None, field=DISTANCE_FIELD):
        self.coefs = coefs or {}          # {W: {"w": float, "b": float}}
        self.field = field

    # ---------- fitting ----------

    def fit(self, distances, targets, windows=(10, 20, 30, 50)):
        """distances: (n_seeds, n_episodes). targets: (n_seeds,) final success rate."""
        from sklearn.linear_model import LogisticRegression

        distances = np.asarray(distances, dtype=float)
        poor = (np.asarray(targets) < POOR_THRESHOLD).astype(int)
        if poor.sum() < 5 or (1 - poor).sum() < 5:
            raise ValueError(f"need both classes present, got {poor.sum()} poor")

        for W in windows:
            x = distances[:, :W].mean(axis=1).reshape(-1, 1)
            lr = LogisticRegression().fit(x, poor)
            self.coefs[str(W)] = {"w": float(lr.coef_[0, 0]),
                                  "b": float(lr.intercept_[0])}
        return self

    # ---------- inference ----------

    def _pick_window(self, n):
        """Use the coefficients fitted at the closest available window."""
        if not self.coefs:
            raise RuntimeError("estimator not fitted -- call fit() or load()")
        return min((int(k) for k in self.coefs), key=lambda W: abs(W - n))

    def skill(self, recent_distances):
        """Point estimate of current skill, in metres."""
        return float(np.mean(recent_distances))

    def p_fail(self, recent_distances):
        """P(this student will not reach competence on their own).

        recent_distances: per-episode distances observed so far, in metres.
        Below ~30 episodes a high value is not trustworthy; see README.
        """
        d = np.asarray(recent_distances, dtype=float)
        if d.size == 0:
            return 0.5
        c = self.coefs[str(self._pick_window(d.size))]
        return float(1.0 / (1.0 + np.exp(-(c["w"] * d.mean() + c["b"]))))

    def n_episodes_fitted(self):
        return sorted(int(k) for k in self.coefs)

    # ---------- persistence ----------

    def save(self, path=PARAMS_PATH):
        Path(path).write_text(json.dumps(
            {"poor_threshold": POOR_THRESHOLD,
             "target_window": list(TARGET_WINDOW),
             "distance_field": self.field,
             "coefs": self.coefs}, indent=2))
        return path

    @classmethod
    def load(cls, path=PARAMS_PATH):
        d = json.loads(Path(path).read_text())
        return cls(coefs=d["coefs"], field=d.get("distance_field", DISTANCE_FIELD))


def build_targets(pool, window=TARGET_WINDOW):
    """Sustained late-training success rate, from the RECORDED goal flags.

    Uses pool[seed].goal rather than thresholding distance -- the recorded flag
    is authoritative, and the two disagree on ~0.1% of episodes.
    """
    a, b = window
    _, goal = stack(pool, "goal")
    return goal[:, a:b].mean(axis=1)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", default=None)
    ap.add_argument("--field", default=DISTANCE_FIELD, choices=["dist", "dist_max"])
    ap.add_argument("--prefer", default="npz", choices=["npz", "monitor"])
    args = ap.parse_args()

    pool = load_pool(args.baseline_dir or BASELINE_DIR, prefer=args.prefer)
    if not pool:
        raise SystemExit(f"no runs found under {args.baseline_dir or BASELINE_DIR}")

    seeds, D = stack(pool, args.field)
    y = build_targets(pool)

    est = SkillEstimator(field=args.field).fit(D, y)
    path = est.save()
    print(f"fitted on {len(seeds)} seeds using '{args.field}', "
          f"{(y < POOR_THRESHOLD).sum()} poor")
    for W in est.n_episodes_fitted():
        c = est.coefs[str(W)]
        print(f"  W={W:<4} w={c['w']:+.4f}  b={c['b']:+.4f}")
    print(f"[SAVE] {path}")


if __name__ == "__main__":
    main()