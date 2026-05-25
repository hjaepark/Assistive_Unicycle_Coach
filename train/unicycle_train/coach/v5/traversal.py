# train/unicycle_train/coach/v5/traversal.py

"""V5 Latent Traversal: change z, see how trajectory changes."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from train.unicycle_train.coach.v5.autoencoder_model import TrajectoryAutoencoder

# ===== Config =====
RUN_DIR = Path("train/unicycle_train/coach/v5/runs/ae_v5_baseline")
VAL_PATH = Path("train/unicycle_train/coach/v5/dataset_processed/val.npz")
EPISODE_IDX = 1000          # which val episode to use as base
DIM_TO_CHANGE = 14           # which latent dim to traverse
VALUES = [-4, -2, 0, 2, 4]  # how much to change (in std units)
FEATURES_TO_PLOT = [0, 8, 16, 19]  # x_pos, wheel_angle, wheel_angvel, act_wheel
FEATURE_NAMES = ['x_pos', 'wheel_angle', 'wheel_angvel', 'act_wheel']

# ===== Load =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(RUN_DIR / "best_model.pt", map_location=device, weights_only=False)
cfg = ckpt['config']
model = TrajectoryAutoencoder(
    feature_dim=21, hidden_dim=cfg['hidden_dim'], latent_dim=cfg['latent_dim'],
    num_layers=cfg['num_layers'], max_seq_len=1500, dropout=0.0,
).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

data = np.load(VAL_PATH)
x = torch.from_numpy(data['features'][EPISODE_IDX]).float().unsqueeze(0).to(device)
m = torch.from_numpy(data['masks'][EPISODE_IDX]).unsqueeze(0).to(device)
T = int(m.sum().item())

# ===== Traverse =====
with torch.no_grad():
    z = model.encode(x, m)  # (1, 32)
    
    trajectories = []
    for v in VALUES:
        z_new = z.clone()
        z_new[0, DIM_TO_CHANGE] = z[0, DIM_TO_CHANGE] + v
        x_recon = model.decode(z_new, T_max=1500)
        trajectories.append(x_recon[0, :T].cpu().numpy())

# ===== Plot =====
fig, axes = plt.subplots(1, len(FEATURES_TO_PLOT), figsize=(16, 4))
colors = plt.cm.coolwarm(np.linspace(0, 1, len(VALUES)))

for col, (feat_idx, feat_name) in enumerate(zip(FEATURES_TO_PLOT, FEATURE_NAMES)):
    ax = axes[col]
    for traj, v, c in zip(trajectories, VALUES, colors):
        ax.plot(traj[:, feat_idx], color=c, label=f'z[{DIM_TO_CHANGE}]+{v}', alpha=0.7)
    ax.set_title(feat_name)
    ax.set_xlabel("timestep")
    if col == 0:
        ax.legend(fontsize=8)

plt.suptitle(f"Traverse latent dim {DIM_TO_CHANGE} (val episode {EPISODE_IDX})")
plt.tight_layout()
out = RUN_DIR / f"traversal_dim{DIM_TO_CHANGE}.png"
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"Saved: {out}")