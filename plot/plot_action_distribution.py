# plot/plot_action_distribution.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

DATA_DIR = '../data'
PARTICIPANTS = [f'P{i:02d}' for i in range(1, 11)]
PHASE = 'training'
FILE_SUFFIX = f'_{PHASE}_trajectory.csv'

HIGH_SUCCESS = ['P02', 'P05', 'P07']
LOW_SUCCESS = ['P04', 'P06', 'P10']

ACTION_COLS = ['final_action_0', 'final_action_1']
ACTION_LABELS = ['Wheel Torque', 'Pitch Torque']

all_data = []

for p_id in PARTICIPANTS:
    file_name = f"{p_id}{FILE_SUFFIX}"
    file_path = os.path.join(DATA_DIR, p_id, file_name)

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            df['participant_id'] = p_id

            if p_id in HIGH_SUCCESS:
                df['group'] = 'High Success'
            elif p_id in LOW_SUCCESS:
                df['group'] = 'Low Success'
            else:
                df['group'] = 'Mid'

            all_data.append(df)
        except Exception as e:
            print(f"Error {p_id}: {e}")

if not all_data:
    print("No data found.")
    exit()

df_all = pd.concat(all_data, ignore_index=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, (col, label) in enumerate(zip(ACTION_COLS, ACTION_LABELS)):
    ax = axes[idx]

    h = ax.hist2d(
        df_all['num_steps'],
        df_all[col],
        bins=[50, 80],
        cmap='YlOrRd',
        cmin=1
    )
    fig.colorbar(h[3], ax=ax, label='Count')

    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(f'Action Distribution: {label}', fontsize=13, pad=10)
    ax.set_ylim(-1.1, 1.1)

    ax.axhline(y=-1.0, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=+1.0, color='blue', linestyle='--', alpha=0.5, linewidth=1)

fig.suptitle('Human Action Distribution Over Training [All Participants]', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('plot_action_heatmap_all.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved plot_action_heatmap_all.png")


fig, axes = plt.subplots(2, 2, figsize=(16, 12))

groups = ['High Success', 'Low Success']

for row, group in enumerate(groups):
    df_group = df_all[df_all['group'] == group]

    for col_idx, (col, label) in enumerate(zip(ACTION_COLS, ACTION_LABELS)):
        ax = axes[row][col_idx]

        h = ax.hist2d(
            df_group['num_steps'],
            df_group[col],
            bins=[50, 80],
            cmap='YlOrRd',
            cmin=1
        )
        fig.colorbar(h[3], ax=ax, label='Count')

        ax.set_xlabel('Timesteps', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{group} — {label}', fontsize=12, pad=8)
        ax.set_ylim(-1.1, 1.1)

        ax.axhline(y=-1.0, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=+1.0, color='blue', linestyle='--', alpha=0.5, linewidth=1)

fig.suptitle('Action Distribution: High vs Low Success Groups [Training]', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('plot_action_heatmap_groups.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved plot_action_heatmap_groups.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (col, label) in enumerate(zip(ACTION_COLS, ACTION_LABELS)):
    ax = axes[idx]

    for group, color in zip(['High Success', 'Low Success'], ['#4c72b0', '#dd8452']):
        df_group = df_all[df_all['group'] == group]
        ax.hist(
            df_group[col],
            bins=100,
            alpha=0.5,
            label=group,
            color=color,
            density=True
        )

    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Action Distribution: {label}', fontsize=13, pad=10)
    ax.legend()
    ax.axvline(x=-1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=+1.0, color='gray', linestyle='--', alpha=0.5)

fig.suptitle('Marginal Action Distribution — High vs Low Success', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('plot_action_marginal.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved plot_action_marginal.png")