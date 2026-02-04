import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

DATA_DIR = '../data'
PARTICIPANTS = [f'P{i:02d}' for i in range(1, 11)]
TARGET_FILE_SUFFIX = '_training_trajectory.csv'
DISTANCE_COL = 'obs_0'

TOTAL_STEPS = 72000
# Divide into 20 bins of 3600 steps each
NUM_BINS = 20 
BIN_SIZE = TOTAL_STEPS / NUM_BINS

all_binned_data = []

for p_id in PARTICIPANTS:
    file_name = f"{p_id}{TARGET_FILE_SUFFIX}"
    file_path = os.path.join(DATA_DIR, p_id, file_name)
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            
            ep_max_dist = df.groupby('episode')[DISTANCE_COL].max()
            ep_end_time = df.groupby('episode')['num_steps'].max()
            
            summary_df = pd.DataFrame({
                'max_dist': ep_max_dist,
                'total_timesteps': ep_end_time
            })
            
            summary_df['time_bin'] = (summary_df['total_timesteps'] // BIN_SIZE).astype(int)
            
            participant_binned = summary_df.groupby('time_bin')['max_dist'].mean().reset_index()
            
            participant_binned['participant_id'] = p_id
            participant_binned['condition'] = 'Without-Coach'
            
            participant_binned['training_progress_timesteps'] = participant_binned['time_bin'] * BIN_SIZE
            
            all_binned_data.append(participant_binned)
            
        except Exception as e:
            print(f"Error {p_id}: {e}")

if not all_binned_data:
    exit()

final_df = pd.concat(all_binned_data, ignore_index=True)


# Plot figure
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=final_df,
    x='training_progress_timesteps',
    y='max_dist',
    hue='condition',
    palette=['#4c72b0'],
    linewidth=2.5,
    marker='o',
    errorbar=('ci', 95)
)

plt.title('Learning Curve [Training Phase]', fontsize=15, pad=15)
plt.xlabel('Timesteps', fontsize=12)
plt.ylabel('Survival Distance (m)', fontsize=12)
plt.xlim(0, TOTAL_STEPS) # Fix x-axis to total steps for Training Phase (72,000)

plt.legend(title=None)

plt.savefig('plot_learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()