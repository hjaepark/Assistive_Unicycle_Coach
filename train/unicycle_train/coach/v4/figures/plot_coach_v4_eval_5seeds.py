import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Held-out student seeds used for V4 evaluation (not seen during coach training)
seeds = [500, 600, 700, 800, 900]

# Storage for the three success rates we need per seed
uncoached = []
v31_coached = []
v4_coached = []

# Pull V4 data from the consolidated eval folder
v4_base = "train/unicycle_train/coach/v4/eval_logs/eval_20260424_150216_pool100_iter100"
for seed in seeds:
    # No coach condition gives us the uncoached baseline for this seed
    with open(f"{v4_base}/no_coach/seed_{seed}/result.json") as f:
        d = json.load(f)
    uncoached.append(d["evaluation"]["success_rate"] * 100)

    # With coach condition gives us the V4-coached result
    with open(f"{v4_base}/with_coach/seed_{seed}/result.json") as f:
        d = json.load(f)
    v4_coached.append(d["evaluation"]["success_rate"] * 100)

# Pull V3.1 data from per-seed eval folders
for seed in seeds:
    # Glob handles the timestamp portion that varies per folder
    path = glob.glob(f"train/unicycle_train/coach/v3/eval_logs/eval_*_seed{seed}/with_coach/summary.json")[0]
    with open(path) as f:
        d = json.load(f)
    v31_coached.append(d["evaluation_aggregate"]["success_rate_mean"] * 100)

# Compute deltas: how much each coach changed success rate vs. uncoached baseline
v31_delta = [v31_coached[i] - uncoached[i] for i in range(len(seeds))]
v4_delta = [v4_coached[i] - uncoached[i] for i in range(len(seeds))]

# Set up the grouped bar chart with two bars per seed
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(seeds))
width = 0.35  # Width of each bar; total group width = 2 * width

# Two bar groups: V3.1 on the left, V4 on the right of each seed position
bars_v31 = ax.bar(x - width/2, v31_delta, width, label="V3.1 Coach", color="#5B9BD5")
bars_v4 = ax.bar(x + width/2, v4_delta, width, label="V4 Coach", color="#ED7D31")

# Horizontal line at 0 separates helpful (positive) from harmful (negative)
ax.axhline(y=0, color="black", linewidth=0.8)

# Simplified y-axis label and no title; the slide structure provides context
ax.set_xlabel("Student Seed", fontsize=12)
ax.set_ylabel("Success Rate Difference", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.legend(loc="lower left")
ax.grid(axis="y", alpha=0.3)

# Annotate each bar with its numeric value for precision
for bars in [bars_v31, bars_v4]:
    for bar in bars:
        height = bar.get_height()
        # Place label above positive bars and below negative bars
        offset = 2 if height >= 0 else -5
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                f"{height:+.0f}", ha="center", fontsize=9)

# Build an absolute path to the figures directory based on this script's location
# This ensures the figure saves in the right place regardless of how the script is invoked
script_dir = Path(__file__).resolve().parent
output_path = script_dir / "v4_delta_plot.png"

plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved {output_path}")