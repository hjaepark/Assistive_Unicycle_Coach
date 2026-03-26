# train/unicycle_train/coach/v1/plot_coach_v0_v1_ablation.py
"""
Coach Ablation Plot: Phase 2A learning curve + Phase 2B post-test success rate.
Usage:
    python -m train.unicycle_train.coach.v1.plot_coach_v0_v1_ablation \
        --no_coach <eval_logs_path> --v0 <eval_logs_path> --v1 <eval_logs_path> \
        --label 72k
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CONDITIONS = {
    "no_coach": ("#888888", "No Coach"),
    "v0":       ("#DA9022", r"V0 (random init, $G=1000$)"),
    "v1":       ("#1F7BE4", r"V1 (BC pretrain, $G=1000$)"),
}


def load_seed_results(eval_dir):
    eval_dir = Path(eval_dir)
    results = []
    for seed_dir in sorted(eval_dir.glob("seed_*")):
        rpath = seed_dir / "result.json"
        if rpath.exists():
            with open(rpath) as f:
                results.append(json.load(f))
    return results


def interpolate_rewards(results, n_points=200):
    max_ts = 0
    for r in results:
        logs = r["training"]["episode_logs"]
        if logs:
            max_ts = max(max_ts, logs[-1]["timestep"])

    ts_grid = np.linspace(0, max_ts, n_points)
    all_curves = []

    for r in results:
        logs = r["training"]["episode_logs"]
        if not logs:
            continue
        timesteps = np.array([e["timestep"] for e in logs])
        rewards = np.array([e["reward"] for e in logs])
        # Rolling average
        w = max(1, len(rewards) // 20)
        padded = np.pad(rewards, (w // 2, w - 1 - w // 2), mode="edge")
        smoothed = np.convolve(padded, np.ones(w) / w, mode="valid")
        # Interpolate onto common grid
        interp = np.interp(ts_grid, timesteps, smoothed)
        all_curves.append(interp)

    all_curves = np.array(all_curves)
    return ts_grid, np.mean(all_curves, axis=0), np.std(all_curves, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_coach", type=str, required=True)
    parser.add_argument("--v0", type=str, required=True)
    parser.add_argument("--v1", type=str, required=True)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    data = {}
    for key, path in [("no_coach", args.no_coach), ("v0", args.v0), ("v1", args.v1)]:
        data[key] = load_seed_results(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for key in ["no_coach", "v0", "v1"]:
        color, label = CONDITIONS[key]
        ts, mean, std = interpolate_rewards(data[key])
        ax1.plot(ts, mean, color=color, linewidth=2, label=label)
        ax1.fill_between(ts, mean - std, mean + std, color=color, alpha=0.10)
    ax1.set_title("Training Phase", fontsize=14)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Episode Reward")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    keys = ["no_coach", "v0", "v1"]
    x_pos = np.arange(len(keys))
    means, stds = [], []
    for key in keys:
        rates = [r["evaluation"]["success_rate"] * 100 for r in data[key]]
        means.append(np.mean(rates))
        stds.append(np.std(rates))

    bars = ax2.bar(x_pos, means, yerr=stds, color=[CONDITIONS[k][0] for k in keys],
                   width=0.6, capsize=5, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([CONDITIONS[k][1] for k in keys], fontsize=10)
    ax2.set_title("Post-test Phase", fontsize=14)
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(0, 105)

    plt.tight_layout(w_pad=3)
    out_dir = Path(__file__).resolve().parent
    suffix = f"_{args.label}" if args.label else ""
    save_path = out_dir / f"plot_coach_v0_v1_ablation{suffix}.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()