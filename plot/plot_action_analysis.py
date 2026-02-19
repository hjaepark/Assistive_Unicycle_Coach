# plot/plot_action_analysis.py

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

# ── Data Loading ──
all_data = []

for p_id in PARTICIPANTS:
    file_name = f"{p_id}{FILE_SUFFIX}"
    file_path = os.path.join(DATA_DIR, p_id, file_name)

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)

            if p_id in HIGH_SUCCESS:
                group = 'High Success'
            elif p_id in LOW_SUCCESS:
                group = 'Low Success'
            else:
                continue

            for ep_id, ep_df in df.groupby('episode'):
                for col, label in zip(ACTION_COLS, ACTION_LABELS):
                    actions = ep_df[col].values

                    # Sign change count
                    signs = np.sign(actions)
                    sign_changes = np.sum(signs[1:] != signs[:-1])
                    sign_change_rate = sign_changes / max(len(actions) - 1, 1)

                    all_data.append({
                        'participant_id': p_id,
                        'group': group,
                        'episode': ep_id,
                        'action': label,
                        'sign_change_rate': sign_change_rate
                    })

        except Exception as e:
            print(f"Error {p_id}: {e}")

if not all_data:
    print("No data found.")
    exit()

df_stats = pd.DataFrame(all_data)

# ── Plot ──
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(8, 6))

ax = sns.boxplot(
    data=df_stats, x='action', y='sign_change_rate',
    hue='group', palette={'High Success': '#4c72b0', 'Low Success': '#dd8452'},
    showfliers=False,
    boxprops=dict(alpha=0.7)
)

plt.title('Sign Change Rate per Episode [Training]', fontsize=14, pad=15)
plt.ylabel('Sign Change Rate', fontsize=12)
plt.xlabel(None)
plt.legend(title=None)

plt.tight_layout()
plt.savefig('plot_action_sign_change.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved plot_action_sign_change.png")