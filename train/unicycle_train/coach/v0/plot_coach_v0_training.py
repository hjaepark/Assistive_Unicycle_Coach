import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


EXPERIMENTS = [
    {"code": "A", "run": "run_20260307_131327", "sim_time_max": 1200.0},
    {"code": "B", "run": "run_20260308_120353", "sim_time_max": 360.0},
    {"code": "C", "run": "run_20260308_195956", "sim_time_max": 1200.0},
]


def load_experiment_df(log_root: Path, meta: dict) -> pd.DataFrame:
    run_dir = log_root / meta["run"]
    log_path = run_dir / "training_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing config/log in {run_dir}")

    with open(log_path, "r") as f:
        log = json.load(f)
    if not log:
        raise ValueError(f"Empty training log: {log_path}")

    df = pd.DataFrame(
        {
            "experiment": meta["code"],
            "student_idx": [int(e["student_idx"]) for e in log],
            "converged": [bool(e["converged"]) for e in log],
            "sim_time": [float(e["sim_time"]) for e in log],
        }
    )
    t_max = float(meta["sim_time_max"])
    df["reward"] = np.where(df["converged"], (1.0 - (df["sim_time"] / t_max)) * 100.0, -100.0)
    df["status"] = np.where(df["converged"], "Converged", "Timeout/Fail")
    return df


def plot_single_figure(df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.6), sharex=False, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    label_fontsize = 10
    title_fontsize = 12

    palette = {"Converged": "#1a9850", "Timeout/Fail": "#d73027"}
    order = ["A", "B", "C"]

    for ax, code in zip(axes, order):
        sub = df[df["experiment"] == code]
        sns.scatterplot(
            data=sub,
            x="student_idx",
            y="reward",
            hue="status",
            hue_order=["Converged", "Timeout/Fail"],
            palette=palette,
            s=38,
            edgecolor="none",
            alpha=0.90,
            legend=False,
            ax=ax,
        )

        ax.axhline(-100.0, linestyle="--", color="0.45", linewidth=1.1, alpha=0.9)
        ax.set_ylim(-105.0, 65.0)
        ax.set_xlim(0.5, max(sub["student_idx"]) + 0.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"Experiment {code}", loc="right", pad=8, fontsize=title_fontsize)
        ax.grid(alpha=0.22)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("0.2")

    axes[-1].set_xlabel("Student Index", fontsize=label_fontsize)
    fig.supylabel("Coach Reward", fontsize=label_fontsize)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6.5, color=palette["Converged"], label="Converged"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6.5, color=palette["Timeout/Fail"], label="Timeout/Fail"),
        plt.Line2D([0], [0], linestyle="--", linewidth=1.1, color="0.45", label="Timeout reward (-100)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase-1 coach reward figure (Experiment A/B/C).")
    parser.add_argument(
        "--log-root",
        type=str,
        default="train/unicycle_train/coach/v0/logs",
        help="Directory containing run_* folders.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).with_suffix(".png")),
        help="Output figure path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    all_df = pd.concat([load_experiment_df(log_root, meta) for meta in EXPERIMENTS], ignore_index=True)
    plot_single_figure(all_df, Path(args.out), dpi=args.dpi)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
