# plot/plot_episode_return.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

DATA_DIR = '../data'
PARTICIPANTS = [f'P{i:02d}' for i in range(1, 11)]
TARGET_FILE_SUFFIX = '_training_trajectory_updated.csv'



all_data = []

for p_id in PARTICIPANTS:
    file_name = f"{p_id}{TARGET_FILE_SUFFIX}"
    file_path = os.path.join(DATA_DIR, p_id, file_name)

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)

            # Compute episode return (sum of rewards per episode)
            ep_return = df.groupby('episode')['reward'].sum().reset_index()
            ep_return.columns = ['episode', 'episode_return']

            ep_return['participant_id'] = p_id
            all_data.append(ep_return)

        except Exception as e:
            print(f"Error {p_id}: {e}")

if not all_data:
    exit()

final_df = pd.concat(all_data, ignore_index=True)

# Plot: individual lines per participant
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(12, 6))

palette = sns.color_palette("tab10", n_colors=len(PARTICIPANTS))

for i, p_id in enumerate(PARTICIPANTS):
    p_df = final_df[final_df['participant_id'] == p_id]
    plt.plot(
        p_df['episode'],
        p_df['episode_return'],
        color=palette[i],
        linewidth=1.0,
        alpha=0.7,
        label=p_id,
    )

plt.title('Episode Return During Training (Human Participants)', fontsize=15, pad=15)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Episode Return', fontsize=12)

plt.legend(
    title='Participant',
    fontsize=9,
    title_fontsize=10,
    loc='upper left',
    ncol=2,
)

plt.tight_layout()
plt.savefig('plot_episode_return.png', dpi=300, bbox_inches='tight')
plt.show()