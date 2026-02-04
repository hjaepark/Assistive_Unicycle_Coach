import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

DATA_DIR = '../data'
PARTICIPANTS = [f'P{i:02d}' for i in range(1, 11)]
PHASES = ['pre-test', 'post-test']
DISTANCE_COL = 'obs_0'
FINISH_LINE_X = 5.0

data_list = []

# Load data

for p_id in PARTICIPANTS:
    for phase in PHASES:
        file_name = f"{p_id}_{phase}_trajectory.csv"
        file_path = os.path.join(DATA_DIR, p_id, file_name)
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Compute max distance per episode
                ep_max_dist = df.groupby('episode')[DISTANCE_COL].max()
                
                # Computer average max distance
                avg_dist = ep_max_dist.mean()
                
                # Compute success rate
                success_count = (ep_max_dist >= FINISH_LINE_X).sum()
                total_episodes = len(ep_max_dist)
                success_rate = (success_count / total_episodes) * 100.0 if total_episodes > 0 else 0.0
                
                data_list.append({
                    'participant_id': p_id,
                    'condition': 'Without-Coach',
                    'phase': phase,
                    'avg_dist': avg_dist,
                    'success_rate': success_rate
                })
                
            except Exception as e:
                print(f"Error {p_id} {phase}: {e}")

df_results = pd.DataFrame(data_list)

# Orient pre/post phases correctly
df_results['phase'] = pd.Categorical(df_results['phase'], categories=PHASES, ordered=True)

def plot_paired_box(data, y_col, y_label, title, filename, y_limit=None):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(7, 6))

    # Box Plot
    sns.boxplot(
        data=data, x='phase', y=y_col,
        hue='condition', palette=['#4c72b0'],
        showfliers=False,
        boxprops=dict(alpha=0.6)
    )

    # Strip Plot
    sns.stripplot(
        data=data, x='phase', y=y_col,
        color='black', alpha=0.6, jitter=False, size=6
    )

    # Paired Lines
    for p_id in PARTICIPANTS:
        p_data = data[data['participant_id'] == p_id]
        if len(p_data) == 2:
            p_data = p_data.sort_values('phase')
            
            plt.plot(
                [0, 1], 
                p_data[y_col].values, 
                color='gray', alpha=0.4, linewidth=1.0, linestyle='-'
            )

    plt.title(title, fontsize=14, pad=15)
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel(None)
    plt.legend(title=None)
    
    if y_limit:
        plt.ylim(y_limit)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {filename}")


# Figure 1: Survival Distance Gain (Box Plot)
plot_paired_box(
    df_results, 
    y_col='avg_dist', 
    y_label='Survival Distance (m)', 
    title='Survival Distance Gain', 
    filename='survival_distance_gain.png',
    y_limit=(-0.2, 5.2)
)

# Figure 2: Success Rate Gain (Box Plot)
plot_paired_box(
    df_results, 
    y_col='success_rate', 
    y_label='Success Rate (%)', 
    title='Success Rate Gain', 
    filename='success_rate_gain.png',
    y_limit=(-5, 105)
)