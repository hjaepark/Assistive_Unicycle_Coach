# train/unicycle_train/coach/v4_2/figures/plot_eval_4action.py

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


LABEL = "V4.2 4action"
THRESHOLD = 0.05  # +/-5pp threshold for helped vs hurt vs neutral

# Paper-friendly, colorblind-safe palette
COLOR_HELPED = "#1f77b4"   # blue
COLOR_HURT = "#ff7f0e"     # orange
COLOR_NEUTRAL = "#bdbdbd"  # gray


def main():
    # Script lives in v4_2/figures/, eval results live in v4_2/eval_logs/
    here = Path(__file__).resolve().parent
    eval_logs = here.parent / "eval_logs"

    # Collect (seed -> success_rate) for both modes across all timestamp dirs
    pattern = f"eval_*4action_iter100*/{{mode}}/seed_*/result.json"
    rates = {"with_coach": {}, "no_coach": {}}
    for mode in rates:
        for path in eval_logs.glob(pattern.format(mode=mode)):
            seed = int(path.parent.name.split("_")[1])
            with open(path) as f:
                rates[mode][seed] = json.load(f)["evaluation"]["success_rate"]

    # Align by seed - same index across both arrays = same student
    seeds = sorted(set(rates["with_coach"]) & set(rates["no_coach"]))
    uncoached = np.array([rates["no_coach"][s] for s in seeds])
    coached = np.array([rates["with_coach"][s] for s in seeds])
    benefit = coached - uncoached

    # Print key stats so we have numbers to cite alongside the figure
    helped = int((benefit > THRESHOLD).sum())
    hurt = int((benefit < -THRESHOLD).sum())
    neutral = len(benefit) - helped - hurt
    corr = np.corrcoef(uncoached, benefit)[0, 1]
    print(f"{LABEL} ({len(seeds)} seeds): "
          f"helped={helped}, hurt={hurt}, neutral={neutral}, "
          f"corr(uncoached, benefit)={corr:+.3f}")

    # Classify each point so colors match the helped/hurt/neutral counts above
    helped_mask = benefit > THRESHOLD
    hurt_mask = benefit < -THRESHOLD
    neutral_mask = ~(helped_mask | hurt_mask)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Diagonal first (background), then neutral, then helped/hurt on top
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2,
            alpha=0.7, label="y = x (no coach effect)")
    ax.scatter(uncoached[neutral_mask], coached[neutral_mask],
               color=COLOR_NEUTRAL, s=55, edgecolors="black", linewidths=0.5,
               label=f"Neutral (n={neutral})")
    ax.scatter(uncoached[helped_mask], coached[helped_mask],
               color=COLOR_HELPED, s=55, edgecolors="black", linewidths=0.5,
               label=f"Helped (n={helped})")
    ax.scatter(uncoached[hurt_mask], coached[hurt_mask],
               color=COLOR_HURT, s=55, edgecolors="black", linewidths=0.5,
               label=f"Hurt (n={hurt})")

    ax.set_xlabel("Uncoached success rate", fontsize=12)
    ax.set_ylabel("Coached success rate", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(here / "plot_eval_4action.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {here / 'plot_eval_4action.png'}")


if __name__ == "__main__":
    main()