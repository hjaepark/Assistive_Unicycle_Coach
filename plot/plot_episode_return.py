# plot/plot_episode_return.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

HUMAN_DATA_DIR = '../data'
RL_DATA_DIR = '../train/unicycle_train/student/logs_student'
RL_RUN_PREFIX = 'run_20260220_10'  # Filter for this batch of runs

PARTICIPANTS = [f'P{i:02d}' for i in range(1, 11)]
HUMAN_FILE_SUFFIX = '_training_trajectory_updated.csv'

BIN_SIZE = 10  # Number of episodes per bin

human_data = []
human_max_episodes = []

for p_id in PARTICIPANTS:
    file_name = f"{p_id}{HUMAN_FILE_SUFFIX}"
    file_path = os.path.join(HUMAN_DATA_DIR, p_id, file_name)

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            ep_return = df.groupby('episode')['reward'].sum().reset_index()
            ep_return.columns = ['episode', 'episode_return']
            human_max_episodes.append(ep_return['episode'].max())

            ep_return['episode_bin'] = (ep_return['episode'] - 1) // BIN_SIZE
            binned = ep_return.groupby('episode_bin')['episode_return'].mean().reset_index()
            binned['episode_bin_center'] = (binned['episode_bin'] * BIN_SIZE) + (BIN_SIZE / 2)
            binned['source_id'] = p_id
            binned['source_type'] = 'human'
            human_data.append(binned)
        except Exception as e:
            print(f"Error {p_id}: {e}")

# Truncate RL to minimum human episode count for fair comparison
MAX_HUMAN_EPISODES = max(human_max_episodes) if human_max_episodes else 142
print(f"[info] Max human episodes: {MAX_HUMAN_EPISODES}")
print(f"[info] Truncating RL data to {MAX_HUMAN_EPISODES} episodes")

# Load RL data
rl_data = []
rl_runs = sorted(glob.glob(os.path.join(RL_DATA_DIR, f'{RL_RUN_PREFIX}*')))
print(f"[info] Found {len(rl_runs)} RL runs")

for i, run_dir in enumerate(rl_runs):
    traj_path = os.path.join(run_dir, 'trajectory.csv')
    if os.path.exists(traj_path):
        try:
            df = pd.read_csv(traj_path)
            ep_return = df.groupby('episode')['reward'].sum().reset_index()
            ep_return.columns = ['episode', 'episode_return']

            # Truncate to human episode range
            ep_return = ep_return[ep_return['episode'] <= MAX_HUMAN_EPISODES]

            ep_return['episode_bin'] = (ep_return['episode'] - 1) // BIN_SIZE
            binned = ep_return.groupby('episode_bin')['episode_return'].mean().reset_index()
            binned['episode_bin_center'] = (binned['episode_bin'] * BIN_SIZE) + (BIN_SIZE / 2)
            binned['source_id'] = f'RL_{i+1:02d}'
            binned['source_type'] = 'rl'
            rl_data.append(binned)
        except Exception as e:
            print(f"Error {run_dir}: {e}")

# Combine and plot
all_human = pd.concat(human_data, ignore_index=True) if human_data else pd.DataFrame()
all_rl = pd.concat(rl_data, ignore_index=True) if rl_data else pd.DataFrame()

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# Human participants
ax = axes[0]
palette_human = sns.color_palette("tab10", n_colors=len(PARTICIPANTS))

for i, p_id in enumerate(PARTICIPANTS):
    p_df = all_human[all_human['source_id'] == p_id]
    ax.plot(
        p_df['episode_bin_center'], p_df['episode_return'],
        color=palette_human[i], linewidth=1.5, alpha=0.8,
        marker='o', markersize=3, label=p_id,
    )

ax.set_title('Human Participants', fontsize=14, pad=10)
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel(f'Average Return (bin={BIN_SIZE})', fontsize=12)
ax.set_xlim(0, MAX_HUMAN_EPISODES)
ax.legend(fontsize=8, ncol=2, loc='upper left')

# RL agents
ax = axes[1]
palette_rl = sns.color_palette("husl", n_colors=len(rl_runs))

for i, rl_id in enumerate([f'RL_{j+1:02d}' for j in range(len(rl_runs))]):
    r_df = all_rl[all_rl['source_id'] == rl_id]
    ax.plot(
        r_df['episode_bin_center'], r_df['episode_return'],
        color=palette_rl[i], linewidth=1.5, alpha=0.8,
        marker='o', markersize=3, label=rl_id,
    )

ax.set_title('RL Student Agents', fontsize=14, pad=10)
ax.set_xlabel('Episode', fontsize=12)
ax.set_xlim(0, MAX_HUMAN_EPISODES)
ax.legend(fontsize=8, ncol=2, loc='upper left')

#fig.suptitle('Episode Return: Human vs RL Student', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('plot_episode_return.png', dpi=300, bbox_inches='tight')
plt.show()