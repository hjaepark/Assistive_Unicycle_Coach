import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = '../data'
PARTICIPANTS = [f'P{i:02d}' for i in range(1, 11)]
TARGET_FILE_SUFFIX = '_training_trajectory.csv'
REWARD_COL = 'reward' 

TOTAL_STEPS = 72000
NUM_BINS = 20
BIN_SIZE = TOTAL_STEPS / NUM_BINS

all_binned_data = []

for p_id in PARTICIPANTS:
    file_name = f"{p_id}{TARGET_FILE_SUFFIX}"
    file_path = os.path.join(DATA_DIR, p_id, file_name)
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            
            ep_return = df.groupby('episode')[REWARD_COL].sum()
            ep_end_time = df.groupby('episode')['num_steps'].max()
            
            summary_df = pd.DataFrame({
                'episode_return': ep_return,
                'total_timesteps': ep_end_time
            })
            
            summary_df['time_bin'] = (summary_df['total_timesteps'] // BIN_SIZE).astype(int)
            participant_binned = summary_df.groupby('time_bin')['episode_return'].mean().reset_index()
            
            participant_binned['participant_id'] = p_id
            participant_binned['training_progress_timesteps'] = participant_binned['time_bin'] * BIN_SIZE
            
            all_binned_data.append(participant_binned)
            
        except Exception as e:
            print(f"Error {p_id}: {e}")

if not all_binned_data:
    exit()

final_df = pd.concat(all_binned_data, ignore_index=True)


sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

group1_ids = PARTICIPANTS[:5]  # P01 ~ P05
group2_ids = PARTICIPANTS[5:]  # P06 ~ P10

groups = [group1_ids, group2_ids]
titles = ['Participants P01 - P05', 'Participants P06 - P10']

palette = sns.color_palette("tab10", 10)

for i, ax in enumerate(axes):
    subset_df = final_df[final_df['participant_id'].isin(groups[i])]
    
    current_palette = palette[:5] if i == 0 else palette[5:]
    
    sns.lineplot(
        data=subset_df,
        x='training_progress_timesteps',
        y='episode_return',
        hue='participant_id',
        palette=current_palette, 
        linewidth=2.0,
        alpha=0.8,
        marker='o',
        markersize=6,
        ax=ax
    )
    
    ax.set_title(titles[i], fontsize=14, pad=10)
    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_xlim(0, TOTAL_STEPS)
    
    if i == 0:
        ax.set_ylabel('Episode Return', fontsize=12)
    else:
        ax.set_ylabel('')

    ax.legend(title='ID', loc='upper left')

plt.suptitle('Individual Reward Trajectories (Split View)', fontsize=16, y=1.02)

plt.tight_layout()
plt.savefig('plot_individual_reward_split.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved plot_individual_reward_split.png")