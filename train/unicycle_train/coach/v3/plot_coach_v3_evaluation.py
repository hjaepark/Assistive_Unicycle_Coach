#!/usr/bin/env python3
"""
Plot Coach V3 Evaluation Results

Generates: (a) Training learning curve, (b) Post-test success rate bar chart.
Accepts 2 or 3 conditions for flexible comparison.

Usage:
    # 2 conditions:
    python -m train.unicycle_train.coach.v3.plot_coach_v3_evaluation \
        --no_coach <eval_dir>/no_coach \
        --old_coach <eval_dir>/with_coach

    # 3 conditions:
    python -m train.unicycle_train.coach.v3.plot_coach_v3_evaluation \
        --no_coach <eval_dir>/no_coach \
        --old_coach <eval_dir>/with_coach \
        --new_coach <eval_dir>/with_coach
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
    "no_coach":   ("#888888", "No Coach"),
    "old_coach":  ("#DA9022", "V3"),
    "new_coach":  ("#1F7BE4", "V3.1"),
}


def load_seed_results(eval_dir):
    """Load all seed results from an eval directory."""
    eval_dir = Path(eval_dir)
    results = []
    for seed_dir in sorted(eval_dir.glob("seed_*")):
        rpath = seed_dir / "result.json"
        if rpath.exists():
            with open(rpath) as f:
                results.append(json.load(f))
    return results


def interpolate_rewards(results, n_points=200):
    """Interpolate training episode rewards onto common episode grid."""
    max_ep = 0
    for r in results:
        logs = r["training"]["episode_logs"]
        if logs:
            max_ep = max(max_ep, len(logs))

    ep_grid = np.linspace(1, max_ep, n_points)
    all_curves = []

    for r in results:
        logs = r["training"]["episode_logs"]
        if not logs:
            continue
        episodes = np.arange(1, len(logs) + 1)
        rewards = np.array([e["reward"] for e in logs])

        # Rolling average
        w = max(1, len(rewards) // 20)
        padded = np.pad(rewards, (w // 2, w - 1 - w // 2), mode="edge")
        smoothed = np.convolve(padded, np.ones(w) / w, mode="valid")

        # Interpolate onto common grid
        interp = np.interp(ep_grid, episodes, smoothed)
        all_curves.append(interp)

    all_curves = np.array(all_curves)
    return ep_grid, np.mean(all_curves, axis=0), np.std(all_curves, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Plot Coach V3 Evaluation")
    parser.add_argument("--no_coach", type=str, required=True,
                        help="Path to no_coach eval directory (containing seed_* folders)")
    parser.add_argument("--old_coach", type=str, required=True,
                        help="Path to old coach (update OFF) eval directory")
    parser.add_argument("--new_coach", type=str, default=None,
                        help="Path to new coach (update ON) eval directory (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: v3 directory)")
    args = parser.parse_args()

    # Determine which conditions to plot
    keys = ["no_coach", "old_coach"]
    paths = {"no_coach": args.no_coach, "old_coach": args.old_coach}
    if args.new_coach:
        keys.append("new_coach")
        paths["new_coach"] = args.new_coach

    # Load data
    data = {}
    for key in keys:
        data[key] = load_seed_results(paths[key])
        n_seeds = len(data[key])
        print(f"  Loaded {key}: {n_seeds} seeds from {paths[key]}")

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Training Phase Learning Curve
    for key in keys:
        color, label = CONDITIONS[key]
        ep, mean, std = interpolate_rewards(data[key])
        ax1.plot(ep, mean, color=color, linewidth=2, label=label)
        ax1.fill_between(ep, mean - std, mean + std, color=color, alpha=0.10)

    ax1.set_title("Training Phase", fontsize=14)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    # 2. Post-test Success Rate
    x_pos = np.arange(len(keys))
    means, stds = [], []
    colors = []
    labels = []

    for key in keys:
        rates = [r["evaluation"]["success_rate"] * 100 for r in data[key]]
        means.append(np.mean(rates))
        stds.append(np.std(rates))
        colors.append(CONDITIONS[key][0])
        labels.append(CONDITIONS[key][1])

    bars = ax2.bar(x_pos, means, yerr=stds, color=colors,
                   width=0.6, capsize=5, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("Post-test Phase", fontsize=14)
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(0, 105)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 2,
                 f"{mean:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout(w_pad=3)

    suffix = f"_{len(keys)}cond"
    save_path = output_dir / f"plot_coach_v3_evaluation{suffix}.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()