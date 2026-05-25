# train/unicycle_train/coach/v5/pca.py

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from train.unicycle_train.coach.v5.autoencoder_model import TrajectoryAutoencoder
from train.unicycle_train.coach.v5.train_autoencoder import TrajectoryDataset

# ===== Config =====
RUN_DIR = Path("train/unicycle_train/coach/v5/runs/ae_v5_baseline")
VAL_PATH = Path("train/unicycle_train/coach/v5/dataset_processed/val.npz")
RAW_DIR = Path("train/unicycle_train/coach/v5/dataset")
N_SAMPLES = 5000

# ===== Load model =====
device = torch.device("cpu")
ckpt = torch.load(RUN_DIR / "best_model.pt", map_location=device, weights_only=False)
cfg = ckpt['config']
model = TrajectoryAutoencoder(
    feature_dim=21, hidden_dim=cfg['hidden_dim'], latent_dim=cfg['latent_dim'],
    num_layers=cfg['num_layers'], max_seq_len=1500, dropout=0.0,
).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# ===== Encode all val episodes =====
ds = TrajectoryDataset(VAL_PATH)
loader = DataLoader(ds, batch_size=128, num_workers=2)
zs = []
with torch.no_grad():
    for x, m in loader:
        zs.append(model.encode(x, m).numpy())
z_all = np.concatenate(zs)
print(f"z shape: {z_all.shape}")

# ===== PCA =====
pca = PCA(n_components=2)
z_2d = pca.fit_transform(z_all)
print(f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")

# ===== Load metadata =====
data = np.load(VAL_PATH)
ep_ids = data['ep_ids']
seeds = data['seeds']
lengths = data['masks'].sum(axis=1)

# ===== Subsample =====
idx = np.random.default_rng(42).choice(len(z_2d), size=N_SAMPLES, replace=False)
z_plot = z_2d[idx]

# ===== Load real success from raw npz (only for subsampled episodes) =====
print(f"Loading real success labels for {N_SAMPLES} episodes...")
success = np.zeros(N_SAMPLES, dtype=np.int32)
for i, k in enumerate(idx):
    ep = np.load(RAW_DIR / f"student_{seeds[k]:04d}" / f"episode_{ep_ids[k]:06d}.npz")
    success[i] = int(ep['success'])
    if i % 1000 == 0:
        print(f"  {i}/{N_SAMPLES}")

# ===== Plot: 2x2 grid =====
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

configs = [
    (ep_ids[idx],  "episode index",  "viridis"),
    (lengths[idx], "episode length", "plasma"),
    (seeds[idx],   "student seed",   "tab20"),
    (success,      "success (real)", "coolwarm"),
]

for ax, (c, label, cmap) in zip(axes.flat, configs):
    sc = ax.scatter(z_plot[:, 0], z_plot[:, 1], c=c, cmap=cmap, s=4, alpha=0.6)
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA colored by {label}")

plt.tight_layout()
out = RUN_DIR / "pca_4panel.png"
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"Saved: {out}")