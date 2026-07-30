# train/utils/plot_reward.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LABEL_SIZE = 9
SMOOTH_WINDOW = 20


def find_run_dir(root, seed):
    matches = sorted(root.glob(f"seed{seed}_*"))
    return matches[-1] if matches else None


def load_rewards(run_dir):
    df = pd.read_csv(run_dir / "monitor.monitor.csv", skiprows=1)
    return df["r"].to_numpy()


def smooth(x, window=SMOOTH_WINDOW):
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()


def plot_curve(ax, y, color):
    x = range(1, len(y) + 1)
    ax.plot(x, y, color=color, linewidth=0.7, alpha=0.25)
    ax.plot(x, smooth(y), color=color, linewidth=1.2)


def plot_one(ax_pre, ax_base, seed, pretrain_root, baseline_root):
    pre_dir = find_run_dir(pretrain_root, seed)
    base_dir = find_run_dir(baseline_root, seed)

    if pre_dir is None or base_dir is None:
        for ax in (ax_pre, ax_base):
            ax.text(0.5, 0.5, f"seed {seed} (missing)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        return

    plot_curve(ax_pre, load_rewards(pre_dir), "steelblue")
    plot_curve(ax_base, load_rewards(base_dir), "darkorange")

    ax_pre.text(0.01, 0.85, f"seed {seed}",
                transform=ax_pre.transAxes, fontsize=9, fontfamily="monospace")

    for ax in (ax_pre, ax_base):
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent / "figures")
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--seed-end", type=int, default=1099)
    p.add_argument("--per-file", type=int, default=10)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_end + 1))

    for i in range(0, len(seeds), args.per_file):
        chunk = seeds[i : i + args.per_file]
        fig, axes = plt.subplots(len(chunk), 2,
                                 figsize=(10, 1.4 * len(chunk)),
                                 sharey=True,
                                 gridspec_kw={"width_ratios": [4, 3]})
        if len(chunk) == 1:
            axes = [axes]

        for row, seed in zip(axes, chunk):
            ax_pre, ax_base = row
            plot_one(ax_pre, ax_base, seed, args.pretrain_dir, args.baseline_dir)

        axes[0][0].set_title("pretrain", fontsize=LABEL_SIZE)
        axes[0][1].set_title("train", fontsize=LABEL_SIZE)
        axes[-1][0].set_xlabel("episode", fontsize=LABEL_SIZE)
        axes[-1][1].set_xlabel("episode", fontsize=LABEL_SIZE)
        fig.supylabel("reward", fontsize=LABEL_SIZE)
        fig.tight_layout()

        out = args.output_dir / f"reward_seed{chunk[0]}-{chunk[-1]}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()