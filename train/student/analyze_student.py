import argparse
import re
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent / "data"
STUDENT_DIR = BASE / "student"
BASELINE_DIR = BASE / "baseline"

PRETRAIN_EVAL = "eval_ep0400.npz"
BASELINE_EVAL = "eval_ep0300.npz"

PREDICTORS = [
    ("pretrained_final_x_mean", "pre-trained mean final_x"),
    ("pretrained_success",      "pre-trained success rate"),
    ("pretrained_return_mean",  "pre-trained mean return"),
]

OUTCOMES = [
    ("trained_final_x_mean", "trained mean final_x"),
    ("trained_success",      "trained success rate"),
    ("trained_return_mean",  "trained mean return"),
]

def parse_seed(run_dir):
    m = re.search(r"seed(\d+)", run_dir.name)
    return int(m.group(1)) if m else None

def collect(root, eval_name):
    out, dupes = {}, {}
    for d in sorted(p for p in root.glob("seed*") if p.is_dir()):
        seed = parse_seed(d)
        if seed is None:
            continue
        dupes.setdefault(seed, []).append(d)
        f = d / eval_name
        if f.exists():
            out[seed] = f
    for seed, dirs in dupes.items():
        if len(dirs) > 1:
            print(f"[warn] seed {seed} has {len(dirs)} run dirs -> using {out.get(seed, 'NONE')}")
    return out

def metrics(npz_path, prefix):
    d = np.load(npz_path)
    succ = d["successes"].astype(bool)
    fx = d["final_x"].astype(float)
    ret = d["returns"].astype(float)

    return {
        f"{prefix}_success": succ.mean(),
        f"{prefix}_final_x_mean": fx.mean(),
        f"{prefix}_return_mean": ret.mean(),
        f"{prefix}_n_eval": succ.size,
    }

def rank(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    return r

def spearman(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return np.nan
    xr, yr = rank(x[ok]), rank(y[ok])
    xs, ys = xr.std(), yr.std()
    if xs == 0 or ys == 0:
        return np.nan
    return float(((xr - xr.mean()) * (yr - yr.mean())).mean() / (xs * ys))

def auc(x, y):
    ok = ~np.isnan(x)
    xo, yo = x[ok], y[ok]
    n1, n0 = int(yo.sum()), int((~yo).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.empty(len(xo), dtype=float)
    r[np.argsort(xo, kind="mergesort")] = np.arange(1, len(xo) + 1)
    return float((r[yo].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", type=str, default=str(BASE))
    args = ap.parse_args()

    pre = collect(STUDENT_DIR, PRETRAIN_EVAL)
    post = collect(BASELINE_DIR, BASELINE_EVAL)

    seeds = sorted(set(pre) & set(post))
    print(f"[join] pre-trained={len(pre)} trained={len(post)} matched={len(seeds)}")
    for s in sorted(set(pre) ^ set(post)):
        side = "trained" if s in pre else "pre-trained"
        print(f"[warn] seed {s} missing on {side} side")
    if not seeds:
        return

    rows = []
    for s in seeds:
        row = {"seed": s}
        row.update(metrics(pre[s], "pretrained"))
        row.update(metrics(post[s], "trained"))
        rows.append(row)

    cols = list(rows[0].keys())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "join_pretrained_vs_trained.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]}" for c in cols) + "\n")
    print(f"[join] wrote {csv_path}  ({len(rows)} seeds)")

    col = {c: np.array([r[c] for r in rows], dtype=float) for c in cols}

    print("\n[spearman] pre-trained metric vs trained outcome")
    head = " " * 26 + "".join(f"{lbl:>24}" for _, lbl in OUTCOMES)
    print(head)
    for xkey, xlbl in PREDICTORS:
        line = f"{xlbl:<26}"
        for ykey, _ in OUTCOMES:
            line += f"{spearman(col[xkey], col[ykey]):>24.3f}"
        print(line)

    y = col["trained_success"] >= 0.90
    print(f"\n[outcome] {int(y.sum())}/{len(y)} seeds reached >=0.90 after training")
    if 0 < y.sum() < len(y):
        print(f"\n{'predictor':<26} {'AUC':>7}")
        for xkey, xlbl in PREDICTORS:
            print(f"{xlbl:<26} {auc(col[xkey], y):>7.3f}")
        print("\n0.50 = coin flip. <0.50 means inverted, not useless.")

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(OUTCOMES), len(PREDICTORS),
                             figsize=(4.6 * len(PREDICTORS), 4.2 * len(OUTCOMES)),
                             squeeze=False)
    for i, (ykey, ylbl) in enumerate(OUTCOMES):
        for j, (xkey, xlbl) in enumerate(PREDICTORS):
            ax = axes[i][j]
            ax.scatter(col[xkey], col[ykey], s=26, alpha=0.75, edgecolors="none")
            ax.set_xlabel(xlbl)
            ax.set_ylabel(ylbl)
            rho = spearman(col[xkey], col[ykey])
            ax.set_title(f"rho = {rho:.3f}", fontsize=10)
            if ykey.endswith("success"):
                ax.axhline(0.90, ls="--", lw=0.8, c="gray")
                ax.axhline(0.10, ls="--", lw=0.8, c="gray")
                ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    png = out_dir / "scatter_pretrained_vs_trained.png"
    fig.savefig(png, dpi=140)
    print(f"[join] wrote {png}")

if __name__ == "__main__":
    main()