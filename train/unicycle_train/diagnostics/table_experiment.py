# train/unicycle_train/diagnostics/table_experiment.py
#
# One comparison table: every condition vs the solo baseline, paired per seed.
# Metric: per-seed deterministic eval success of the final model (eval.npz).
#
# Usage (add more --exp pairs as new conditions finish):
#   python -m train.unicycle_train.diagnostics.table_experiment \
#       --baseline data/student_baseline \
#       --exp mixture data/student_mixture_actor \
#       --exp demoevery5 data/student_demo_inject

import argparse
import re
from pathlib import Path

import numpy as np
from scipy import stats


def evals(exp_dir: Path):
    latest = {}
    for d in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        m = re.search(r"seed(\d+)", d.name)
        if m:
            latest[int(m.group(1))] = d
    return {s: float(np.load(d / "eval.npz")["successes"].mean())
            for s, d in latest.items() if (d / "eval.npz").exists()}


def row(label, base, treat):
    seeds = sorted(set(base) & set(treat))
    b = np.array([base[s] for s in seeds])
    t = np.array([treat[s] for s in seeds])
    d = t - b
    p = stats.ttest_rel(t, b).pvalue
    strata = {
        "rescue": b == 0.0,
        "middle": (b > 0) & (b < 0.5),
        "strong": b >= 0.5,
    }
    cells = [label, f"{len(seeds)}", f"{t.mean():.3f}", f"{d.mean():+.3f}",
             f"{p:.1e}", f"{(d > 0).sum()}/{(d < 0).sum()}/{(d == 0).sum()}"]
    cells += [f"{d[m].mean():+.3f}" if m.sum() else "-" for m in strata.values()]
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--exp", nargs=2, action="append", metavar=("LABEL", "DIR"),
                    required=True)
    args = ap.parse_args()

    base = evals(args.baseline)
    header = ["condition", "n", "mean", "diff", "p", "+/-/=",
              "d(rescue)", "d(middle)", "d(strong)"]
    rows = [["baseline", f"{len(base)}", f"{np.mean(list(base.values())):.3f}",
             "-", "-", "-", "-", "-", "-"]]
    rows += [row(label, base, evals(Path(d))) for label, d in args.exp]

    widths = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))


if __name__ == "__main__":
    main()