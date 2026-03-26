# train/unicycle_train/coach/v1/plot_coach_v1_evaluation.py

"""
Coach V1 Evaluation Plot
Compares: V0 (random init) vs V1 G=100 (BC) vs V1 G=1000 (BC)

Usage:
    python -m train.unicycle_train.coach.v1.plot_coach_v1_evaluation \
        --v0c path/to/v0c/training_log.json \
        --v1_gs100 path/to/v1_gs100/training_log.json \
        --v1_gs1000 path/to/v1_gs1000/training_log.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
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
    "v0c":        ("#4A90D9", r"V0 (random init, $G=100$)",  "--"),
    "v0_gs1000":  ("#4A90D9", r"V0 (random init, $G=1000$)", "-"),
    "v1_gs100":   ("#E85D3A", r"V1 (BC pretrain, $G=100$)",  "--"),
    "v1_gs1000":  ("#E85D3A", r"V1 (BC pretrain, $G=1000$)", "-"),
}


def smooth(vals, w=5):
    arr = np.array(vals, dtype=float)
    padded = np.pad(arr, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(padded, np.ones(w) / w, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0c", type=str, required=True)
    parser.add_argument("--v0_gs1000", type=str, required=True)
    parser.add_argument("--v1_gs100", type=str, required=True)
    parser.add_argument("--v1_gs1000", type=str, required=True)
    args = parser.parse_args()

    data = {}
    for key, path in [("v0c", args.v0c), ("v0_gs1000", args.v0_gs1000), ("v1_gs100", args.v1_gs100), ("v1_gs1000", args.v1_gs1000)]:
        with open(path) as f:
            data[key] = json.load(f)

    print("=" * 60)
    for key, entries in data.items():
        n = len(entries)
        conv = sum(1 for e in entries if e["converged"])
        print(f"  {CONDITIONS[key][1]}: {conv}/{n} converged ({100*conv/n:.1f}%)")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    for key, entries in data.items():
        color, label, ls = CONDITIONS[key]
        x = [e["student_idx"] for e in entries]
        y = [e["coach_reward"] for e in entries]
        ax1.scatter(x, y, color=color, alpha=0.15, s=15)
        ax1.plot(x, smooth(y), color=color, linewidth=2.2, linestyle=ls, label=label)

    ax1.set_xlabel("Student Index")
    ax1.set_ylabel("Coach Reward")
    ax1.set_title("Coach Reward")
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax1.legend(loc="lower right", framealpha=0.9)

    for key, entries in data.items():
        color, label, ls = CONDITIONS[key]
        x = [e["student_idx"] for e in entries]
        y = [e["sim_time"] for e in entries]
        ax2.scatter(x, y, color=color, alpha=0.15, s=15)
        ax2.plot(x, smooth(y), color=color, linewidth=2.2, linestyle=ls)
    ax2.set_xlabel("Student Index")
    ax2.set_ylabel("Convergence Time (s)")
    ax2.set_title("Student Convergence Time")

    plt.tight_layout(w_pad=3)

    out_dir = Path(__file__).resolve().parent
    save_path = out_dir / "plot_coach_v1_evaluation.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()