# train/unicycle_train/coach/v2/plot_coach_v2_evaluation.py

"""
Coach V2 Evaluation Plot: Lambda sweep comparison.
(a) Mean coach action over student index
(b) Coach reward over student index
(c) Convergence rate bar chart

Usage:
    python -m train.unicycle_train.coach.v2.plot_coach_v2_evaluation \
        --lambda05 <path/to/training_log.json> \
        --lambda25 <path/to/training_log.json> \
        --lambda100 <path/to/training_log.json>
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
    "lambda05":  ("#888888", r"$\lambda_a = 0.5$"),
    "lambda25":  ("#DA9022", r"$\lambda_a = 25$"),
    "lambda100": ("#1F7BE4", r"$\lambda_a = 100$"),
}


def load_log(path):
    with open(path) as f:
        return json.load(f)


def smooth(values, window=5):
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda05", type=str, required=True)
    parser.add_argument("--lambda25", type=str, required=True)
    parser.add_argument("--lambda100", type=str, required=True)
    args = parser.parse_args()

    data = {
        "lambda05": load_log(args.lambda05),
        "lambda25": load_log(args.lambda25),
        "lambda100": load_log(args.lambda100),
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    for key in ["lambda05", "lambda25", "lambda100"]:
        color, label = CONDITIONS[key]
        entries = data[key]
        x = [e["student_idx"] for e in entries]
        y = [e["mean_coach_action"] for e in entries]
        ax1.scatter(x, y, color=color, alpha=0.15, s=15)
        ax1.plot(x, smooth(y), color=color, linewidth=2, label=label)

    ax1.set_xlabel("Student Index")
    ax1.set_ylabel("Mean Coach Action")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_ylim(0.6, 1)

    for key in ["lambda05", "lambda25", "lambda100"]:
        color, label = CONDITIONS[key]
        entries = data[key]
        x = [e["student_idx"] for e in entries]
        y = [e["coach_reward"] for e in entries]
        ax2.scatter(x, y, color=color, alpha=0.15, s=15)
        ax2.plot(x, smooth(y), color=color, linewidth=2, label=label)

    ax2.set_xlabel("Student Index")
    ax2.set_ylabel("Coach Reward")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    keys = ["lambda05", "lambda25", "lambda100"]
    labels = [CONDITIONS[k][1] for k in keys]
    colors = [CONDITIONS[k][0] for k in keys]
    x_pos = np.arange(len(keys))

    conv_rates = []
    for key in keys:
        entries = data[key]
        n_converged = sum(1 for e in entries if e["converged"])
        conv_rates.append(100 * n_converged / len(entries))

    ax3.bar(x_pos, conv_rates, color=colors, width=0.6,
            edgecolor="black", linewidth=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.set_ylabel("Convergence Rate (%)")
    ax3.set_ylim(0, 105)

    plt.tight_layout(w_pad=3)

    out_dir = Path(__file__).resolve().parent
    save_path = out_dir / "plot_coach_v2_evaluation.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()