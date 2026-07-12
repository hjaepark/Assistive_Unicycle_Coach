# train/unicycle_train/diagnostics/compare_paired.py
#
# Paired per-seed comparison: baseline vs. treatment (demo injection).
#
# Reads eval.npz from each seed's run dir in both directories, matches seeds,
# and reports:
#   - overall paired stats (means, diff, paired t-test, Wilcoxon signed-rank)
#   - win / loss / tie counts
#   - stratified breakdown by baseline strength (rescue / middle / no-harm)
#   - rescue rate: fraction of baseline==0 seeds lifted above 0
#   - corr(baseline success, benefit)  <- watch for the V4-style -0.807 pattern
#
# Usage:
#   python -m train.unicycle_train.diagnostics.compare_paired \
#       train/unicycle_train/diagnostics/data/student_baseline \
#       train/unicycle_train/diagnostics/data/student_demo_inject_A2

import argparse
import re
from pathlib import Path

import numpy as np
from scipy import stats


def collect(eval_root: Path):
    """Return {seed: success_rate}, using the latest run dir per seed."""
    latest = {}
    for run_dir in sorted(eval_root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = re.search(r"seed(\d+)", run_dir.name)
        if m is None or not (run_dir / "eval.npz").exists():
            continue
        seed = int(m.group(1))
        # sorted() + overwrite -> keeps the latest timestamp per seed
        latest[seed] = run_dir
    return {
        seed: float(np.load(d / "eval.npz")["successes"].mean())
        for seed, d in latest.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("treatment_dir", type=Path)
    args = ap.parse_args()

    base = collect(args.baseline_dir)
    treat = collect(args.treatment_dir)
    seeds = sorted(set(base) & set(treat))
    if not seeds:
        raise SystemExit("no common seeds with eval.npz in both directories")

    b = np.array([base[s] for s in seeds])
    t = np.array([treat[s] for s in seeds])
    d = t - b

    print(f"common seeds: {len(seeds)}")
    print(f"baseline  mean: {b.mean():.4f}  std: {b.std():.4f}")
    print(f"treatment mean: {t.mean():.4f}  std: {t.std():.4f}")
    print(f"paired diff mean: {d.mean():+.4f}  std: {d.std():.4f}")

    tt = stats.ttest_rel(t, b)
    print(f"paired t-test: t={tt.statistic:+.3f}, p={tt.pvalue:.5f}")
    nz = d[d != 0]
    if len(nz) > 0:
        w = stats.wilcoxon(nz)
        print(f"wilcoxon (zeros dropped, n={len(nz)}): "
              f"stat={w.statistic:.1f}, p={w.pvalue:.5f}")

    print(f"treatment > baseline: {(d > 0).sum()}/{len(seeds)}")
    print(f"treatment < baseline: {(d < 0).sum()}/{len(seeds)}")
    print(f"treatment == baseline: {(d == 0).sum()}/{len(seeds)}")

    # --- stratified breakdown by baseline strength ---
    strata = [
        ("rescue  (base == 0)", b == 0.0),
        ("middle  (0 < base < 0.5)", (b > 0) & (b < 0.5)),
        ("no-harm (base >= 0.5)", b >= 0.5),
    ]
    print("\nstratified by baseline strength:")
    for label, mask in strata:
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:26s} n=0")
            continue
        print(f"  {label:26s} n={n:3d}  "
              f"base={b[mask].mean():.3f} -> treat={t[mask].mean():.3f}  "
              f"diff={d[mask].mean():+.3f}  "
              f"improved={(d[mask] > 0).sum()}/{n}")

    zeros = b == 0.0
    if zeros.sum() > 0:
        rescued = int(((t > 0) & zeros).sum())
        print(f"\nrescue rate: {rescued}/{int(zeros.sum())} "
              f"baseline-zero seeds lifted above 0")

    # V4 showed corr(uncoached strength, benefit) = -0.807; check if it recurs
    if b.std() > 0 and d.std() > 0:
        r = np.corrcoef(b, d)[0, 1]
        print(f"corr(baseline, benefit): {r:+.3f}")

    # per-seed table for the biggest movers
    order = np.argsort(d)
    print("\nbiggest losses:")
    for i in order[:5]:
        print(f"  seed{seeds[i]:02d}: {b[i]:.2f} -> {t[i]:.2f}  ({d[i]:+.2f})")
    print("biggest gains:")
    for i in order[-5:][::-1]:
        print(f"  seed{seeds[i]:02d}: {b[i]:.2f} -> {t[i]:.2f}  ({d[i]:+.2f})")


if __name__ == "__main__":
    main()