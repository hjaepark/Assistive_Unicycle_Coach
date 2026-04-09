#!/usr/bin/env python3
"""
Plot Coach V3 Training Results

Usage:
    python -m train.unicycle_train.coach.v3.plot_coach_v3_training --log <log.json>
    python -m train.unicycle_train.coach.v3.plot_coach_v3_training --log <log1.json> --log2 <log2.json> --labels "V3" "V3.1"
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_log(path):
    with open(path) as f:
        return json.load(f)


def rolling_mean(values, window=20):
    result = np.full(len(values), np.nan)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start:i + 1])
    return result


def rolling_intervention_rate(data, window=20):
    w_ts = [entry["w_t"] for entry in data]
    result = np.full(len(w_ts), np.nan)
    for i in range(len(w_ts)):
        start = max(0, i - window + 1)
        result[i] = np.mean(w_ts[start:i + 1])
    return result


def get_iterations(data):
    if "iteration" in data[0]:
        return [entry["iteration"] for entry in data]
    elif "round" in data[0]:
        return [entry["round"] for entry in data]
    else:
        return list(range(1, len(data) + 1))


def get_episodes(data):
    if "total_episodes" in data[0]:
        return [entry["total_episodes"] for entry in data]
    else:
        iters = get_iterations(data)
        return [i * 20 for i in iters]


COLOR = "tab:blue"


def plot_single(data, output_dir, x_axis="episodes"):
    x = get_episodes(data) if x_axis == "episodes" else get_iterations(data)
    x_label = "Episodes" if x_axis == "episodes" else "Iteration"

    eval_success = [e["eval_success_rate"] * 100 for e in data]
    intervention_rate = rolling_intervention_rate(data, window=20) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x, rolling_mean(eval_success, 20), color=COLOR, linewidth=2)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Self-Practice Phase Performance")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, intervention_rate, color=COLOR, linewidth=2)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Intervention Rate (%)")
    ax2.set_title("Coach Intervention Rate")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "coach_v3_training.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_comparison(data1, data2, label1, label2, output_dir, x_axis="episodes"):
    x1 = get_episodes(data1) if x_axis == "episodes" else get_iterations(data1)
    x2 = get_episodes(data2) if x_axis == "episodes" else get_iterations(data2)
    x_label = "Episodes" if x_axis == "episodes" else "Iteration"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Self-Practice Phase Performance
    eval1 = [e["eval_success_rate"] * 100 for e in data1]
    eval2 = [e["eval_success_rate"] * 100 for e in data2]
    ax1.plot(x1, rolling_mean(eval1, 20), color='tab:blue', linewidth=2, label=label1)
    ax1.plot(x2, rolling_mean(eval2, 20), color='tab:orange', linewidth=2, label=label2)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Self-Practice Phase Performance")
    ax1.set_ylim(-5, 105)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Coach Intervention Rate
    int1 = rolling_intervention_rate(data1, 20) * 100
    int2 = rolling_intervention_rate(data2, 20) * 100
    ax2.plot(x1, int1, color='tab:blue', linewidth=2, label=label1)
    ax2.plot(x2, int2, color='tab:orange', linewidth=2, label=label2)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Intervention Rate (%)")
    ax2.set_title("Coach Intervention Rate")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "coach_v3_training_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Coach V3 Training Results")
    parser.add_argument("--log", type=str, required=True,
                        help="Path to training_log.json")
    parser.add_argument("--log2", type=str, default=None,
                        help="Path to second training_log.json (for comparison)")
    parser.add_argument("--labels", type=str, nargs=2, default=None,
                        help="Labels for comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: same as log file)")
    parser.add_argument("--x-axis", type=str, default="episodes",
                        choices=["episodes", "iterations"],
                        help="X-axis unit (default: episodes)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        return

    output_dir = Path(args.output) if args.output else log_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data1 = load_log(log_path)
    print(f"Loaded: {log_path} ({len(data1)} iterations)")

    if args.log2:
        log2_path = Path(args.log2)
        if not log2_path.exists():
            print(f"Error: {log2_path} not found")
            return

        data2 = load_log(log2_path)
        print(f"Loaded: {log2_path} ({len(data2)} iterations)")

        labels = args.labels if args.labels else ["Run 1", "Run 2"]
        plot_comparison(data1, data2, labels[0], labels[1], output_dir, x_axis=args.x_axis)
    else:
        plot_single(data1, output_dir, x_axis=args.x_axis)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()