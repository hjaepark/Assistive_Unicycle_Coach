# train/unicycle_train/coach/v0/plot_coach_v0_evaluation.py
#
# Plot Phase 2 evaluation results: No Coach vs Coach A vs Coach C
#
# Usage:
#   python -m train.unicycle_train.coach.v0.plot_coach_v0_evaluation
#   python -m train.unicycle_train.coach.v0.plot_coach_v0_evaluation --eval-root train/unicycle_train/coach/v0/eval_logs

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Configuration: map experiment labels to eval_logs directories
# ============================================================
EXPERIMENTS = [
    {
        "label": "No Coach",
        "dir": "eval_no_coach_20260311_031941",
        "color": "#5B8DBE",
        "linestyle": "-",
    },
    {
        "label": "Coach A",
        "dir": "eval_with_coach_20260311_094739",
        "color": "#E07B54",
        "linestyle": "-",
    },
    {
        "label": "Coach C",
        "dir": "eval_with_coach_20260311_113347",
        "color": "#6BAF6E",
        "linestyle": "-",
    },
]


# ============================================================
# Data loading
# ============================================================
def load_eval_results(eval_root: Path, exp: dict) -> dict:
    """Load all seed results for one experiment condition."""
    exp_dir = eval_root / exp["dir"]
    if not exp_dir.exists():
        raise FileNotFoundError(f"Missing eval directory: {exp_dir}")

    seed_dirs = sorted(exp_dir.glob("seed_*"))
    if not seed_dirs:
        raise ValueError(f"No seed directories found in {exp_dir}")

    all_results = []
    for sd in seed_dirs:
        result_path = sd / "result.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                all_results.append(json.load(f))

    return {
        "label": exp["label"],
        "color": exp["color"],
        "linestyle": exp["linestyle"],
        "results": all_results,
    }


def build_episode_df(loaded: dict, phase: str = "evaluation") -> pd.DataFrame:
    """
    Build a DataFrame of episode-level logs across seeds.
    Columns: seed, episode, reward, goal_reached, is_fallen, x_position, ...
    """
    rows = []
    for result in loaded["results"]:
        seed = result["seed"]
        for ep in result[phase]["episode_logs"]:
            ep_copy = dict(ep)
            ep_copy["seed"] = seed
            ep_copy["condition"] = loaded["label"]
            rows.append(ep_copy)
    return pd.DataFrame(rows)


# ============================================================
# Plot 1: Mean reward over episodes (line graph with std shading)
# ============================================================
def plot_reward_curve(all_loaded: list, phase: str, ax: plt.Axes, max_episodes: int = None):
    """
    Line graph: mean episode reward vs episode index.
    Each condition is a line with shaded std across seeds.
    """
    for loaded in all_loaded:
        df = build_episode_df(loaded, phase)
        if df.empty:
            continue

        # Group by episode number across seeds
        grouped = df.groupby("episode")["reward"]
        means = grouped.mean()
        stds = grouped.std().fillna(0)
        counts = grouped.count()

        # Only plot episodes where at least 2 seeds have data
        mask = counts >= 2
        eps = means.index[mask]
        mu = means.values[mask]
        sd = stds.values[mask]

        if max_episodes:
            clip = eps <= max_episodes
            eps = eps[clip]
            mu = mu[clip]
            sd = sd[clip]

        ax.plot(eps, mu, color=loaded["color"], linestyle=loaded["linestyle"],
                linewidth=1.8, label=loaded["label"], alpha=0.9)
        ax.fill_between(eps, mu - sd, mu + sd, color=loaded["color"], alpha=0.15)

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Episode Reward", fontsize=10)
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.axhline(0, color="0.5", linewidth=0.5, linestyle="--", alpha=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.3")


# ============================================================
# Plot 2: Per-seed final x position (grouped bar chart)
# ============================================================
def plot_per_seed_x_position(all_loaded: list, phase: str, ax: plt.Axes):
    """
    Grouped bar chart: final x position per seed for each condition.
    """
    # Collect data
    conditions = []
    for loaded in all_loaded:
        for result in loaded["results"]:
            conditions.append({
                "condition": loaded["label"],
                "color": loaded["color"],
                "seed": result["seed"],
                "mean_x_position": result[phase]["mean_x_position"],
            })

    df = pd.DataFrame(conditions)
    seeds = sorted(df["seed"].unique())
    n_conditions = len(all_loaded)
    n_seeds = len(seeds)

    width = 0.7 / n_conditions
    x_pos = np.arange(n_seeds)

    for i, loaded in enumerate(all_loaded):
        sub = df[df["condition"] == loaded["label"]].set_index("seed")
        values = [sub.loc[s, "mean_x_position"] if s in sub.index else 0.0 for s in seeds]
        offset = (i - (n_conditions - 1) / 2) * width
        bars = ax.bar(x_pos + offset, values, width,
                      color=loaded["color"], edgecolor="0.3", linewidth=0.6,
                      alpha=0.85, label=loaded["label"])

        # Value labels
        for bar, v in zip(bars, values):
            y = max(bar.get_height(), 0) + 0.08
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Seed", fontsize=10)
    ax.set_ylabel("Mean Final X Position (m)", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.axhline(5.0, color="#d73027", linewidth=1.0, linestyle="--", alpha=0.6, label="Goal (5.0m)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(-0.5, 5.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.3")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Plot Phase 2 evaluation results.")
    parser.add_argument(
        "--eval-root", type=str,
        default="train/unicycle_train/coach/v0/eval_logs",
        help="Root directory containing eval_* folders.",
    )
    parser.add_argument("--out", type=str, default=None, help="Output figure path.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)

    # Load all experiment results
    all_loaded = []
    for exp in EXPERIMENTS:
        try:
            loaded = load_eval_results(eval_root, exp)
            all_loaded.append(loaded)
            n_seeds = len(loaded["results"])
            print(f"[OK] {exp['label']}: {n_seeds} seeds loaded from {exp['dir']}")
        except (FileNotFoundError, ValueError) as e:
            print(f"[SKIP] {exp['label']}: {e}")

    if not all_loaded:
        print("[ERROR] No experiments loaded. Check --eval-root path.")
        return

    # --------------------------------------------------------
    # Figure: 2 rows
    # --------------------------------------------------------
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.2, 1]})

    # Row 1: Evaluation reward curve
    plot_reward_curve(all_loaded, phase="evaluation", ax=axes[0])

    # Row 2: Per-seed x position
    plot_per_seed_x_position(all_loaded, phase="evaluation", ax=axes[1])

    fig.tight_layout(h_pad=3.0)

    # Save
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(__file__).with_name("plot_coach_v0_evaluation.png")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()