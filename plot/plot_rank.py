# plot/plot_rank.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

DATA_DIR = '../data'
PARTICIPANTS = [f'P{i:02d}' for i in range(1, 11)]
PHASE = 'post-test'
FILE_SUFFIX = f'_{PHASE}_trajectory.csv' 

data_list = []

# Data Loading
for p_id in PARTICIPANTS:
    file_name = f"{p_id}{FILE_SUFFIX}"
    file_path = os.path.join(DATA_DIR, p_id, file_name)
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            
            # Calculate Success Rate per Episode
            # 1 if success_indicator occurred at least once in the episode
            episode_success = df.groupby('episode')['success_indicator'].max()
            success_rate = episode_success.mean() * 100.0
            
            data_list.append({
                'participant_id': p_id,
                'success_rate': success_rate
            })
            
        except Exception as e:
            print(f"Error {p_id}: {e}")

if not data_list:
    print("No data found.")
    exit()

df_rank = pd.DataFrame(data_list)
# Sort by success rate descending
df_rank = df_rank.sort_values(by='success_rate', ascending=False)

# Plot Figure (Matched Style)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=df_rank,
    x='participant_id',
    y='success_rate',
    color='#4c72b0'
)

plt.title(f'Participant Success Rate Ranking [{PHASE}]', fontsize=15, pad=15)
plt.xlabel('Participant ID', fontsize=12)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.ylim(0, 110)

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=11)

plt.savefig('plot_rank.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved plot_rank.png")