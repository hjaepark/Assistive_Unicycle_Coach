import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import numpy as np

# Set plot style
plt.style.use('ggplot')
sns.set_context('paper', font_scale=1.2)

# Paths
data_dir = Path(__file__).parent.parent / "data"
conditions = ["with-coach", "without-coach"]
phases = ["pre-test", "training", "post-test"]
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# Load summary data
summary_data = []
for condition in conditions:
    condition_dir = Path(data_dir) / condition
    if not condition_dir.exists():
        print(f"No data found for condition: {condition}")
        continue
    for participant_dir in condition_dir.iterdir():
        if not participant_dir.is_dir():
            continue
        participant_id = participant_dir.name
        for phase in phases:
            summary_file = participant_dir / f"{phase}_summary.csv"
            if summary_file.exists():
                df = pd.read_csv(summary_file)
                summary_data.append(df)
            else:
                print(f"Missing: {summary_file}")

if not summary_data:
    raise ValueError("No summary data found")

df = pd.concat(summary_data, ignore_index=True)

# Define metrics to plot
metrics = [
    ("success_count", "Success Count"),
    ("cumulative_reward", "Cumulative Reward"),
    ("num_failures", "Number of Failures"),
    ("avg_distance_traveled", "Average Distance Traveled")
]

# Generate bar plots for each metric across phases
for metric, ylabel in metrics:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df, x="phase", y=metric, hue="condition", estimator="mean", errorbar="sd", ax=ax)
    ax.set_xlabel("Phase")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} Comparison Across Phases")

    # Add exact numbers on top of bars
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        ax.text(p.get_x() + p.get_width() / 2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=300)
    plt.close()

# Bar Plot: Average Learning Gains (Post-Test - Pre-Test) by Condition
fig, ax = plt.subplots(figsize=(6, 5))
gains = []
for condition in conditions:
    condition_data = df[df["condition"] == condition]
    if condition_data.empty:
        continue
    # Calculate learning gain for each participant in the condition
    participant_gains = []
    for participant in condition_data["participant_id"].unique():
        pre = condition_data[(condition_data["participant_id"] == participant) & (condition_data["phase"] == "pre-test")]["avg_distance_traveled"]
        post = condition_data[(condition_data["participant_id"] == participant) & (condition_data["phase"] == "post-test")]["avg_distance_traveled"]
        if not pre.empty and not post.empty:
            gain = post.iloc[0] - pre.iloc[0]
            participant_gains.append(gain)
    # Compute average learning gain for the condition
    if participant_gains:
        avg_gain = sum(participant_gains) / len(participant_gains)
        gains.append({"condition": condition, "learning_gain": avg_gain})

gains_df = pd.DataFrame(gains)
sns.barplot(data=gains_df, x="condition", y="learning_gain", hue="condition", ax=ax)
ax.set_xlabel("Condition")
ax.set_ylabel("Average Learning Gain (Post-Test - Pre-Test, Distance Traveled)")
ax.set_title("Average Learning Gain Comparison (Distance Traveled)")

# Add exact numbers on top of bars
for p in ax.patches:
    height = p.get_height()
    if np.isnan(height):
        continue
    ax.text(p.get_x() + p.get_width() / 2., height, f'{height:.2f}',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "learning_gain_comparison.png"), dpi=300)
plt.close()