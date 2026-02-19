# train/unicycle_train/coach/estimators/gru_estimator.py

"""
GRU Latent Skill Tracker
=========================
Trains a GRU to estimate latent skill z_k from episode-level performance scores.

Target: future H-episode average of performance_score
    target_k = (1/H) * sum(y_{k+1}, ..., y_{k+H})

Input features per timestep (episode):
    - performance_score
    - termination one-hot (fall, timeout, success)

Usage:
    python train_latent_gru.py --csv <path_to_csv> --H 10 --epochs 50
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 1. DATA PROCESSING
# ============================================================

TERMINATION_MAP = {'fall': 0, 'timeout': 1, 'success': 2}

def load_and_process(csv_path):
    """Load CSV and group by run_id into episode sequences."""
    df = pd.read_csv(csv_path)
    df = df.sort_values(['run_id', 'episode']).reset_index(drop=True)

    # One-hot encode termination
    df['term_fall'] = (df['termination'] == 'fall').astype(float)
    df['term_timeout'] = (df['termination'] == 'timeout').astype(float)
    df['term_success'] = (df['termination'] == 'success').astype(float)

    feature_cols = ['performance_score', 'term_fall', 'term_timeout', 'term_success']

    sequences = {}
    for run_id, group in df.groupby('run_id'):
        sequences[run_id] = {
            'features': group[feature_cols].values.astype(np.float32),
            'scores': group['performance_score'].values.astype(np.float32),
            'episodes': group['episode'].values,
        }

    return sequences, feature_cols


def build_targets(scores, H):
    """
    Build future-H-average targets for each timestep.
    target_k = mean(y_{k+1}, ..., y_{k+H})
    Last H episodes have no valid target -> marked as NaN.
    """
    T = len(scores)
    targets = np.full(T, np.nan, dtype=np.float32)
    for k in range(T - H):
        targets[k] = np.mean(scores[k+1 : k+1+H])
    return targets


# ============================================================
# 2. DATASET
# ============================================================

class EpisodeSequenceDataset(Dataset):
    """Each item is one full run (variable length sequence)."""

    def __init__(self, sequences, H):
        self.data = []
        for run_id, seq in sequences.items():
            targets = build_targets(seq['scores'], H)
            # Mask: True where target is valid
            mask = ~np.isnan(targets)
            targets = np.nan_to_num(targets, nan=0.0)
            self.data.append({
                'run_id': run_id,
                'features': torch.tensor(seq['features']),       # (T, F)
                'targets': torch.tensor(targets),                 # (T,)
                'mask': torch.tensor(mask),                       # (T,)
                'scores': torch.tensor(seq['scores']),            # (T,)
                'episodes': seq['episodes'],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# 3. MODEL
# ============================================================

class LatentGRU(nn.Module):
    """
    GRU-based latent skill tracker.
    Input: episode features (performance_score + termination one-hot)
    Output: predicted future average performance (skill estimate)
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, hidden=None):
        """
        x: (1, T, F) - single run sequence
        Returns:
            preds: (1, T) - skill estimate at each timestep
            hidden: GRU hidden state
            latent: (1, T, hidden_dim) - raw GRU output (latent representation)
        """
        gru_out, hidden = self.gru(x, hidden)  # (1, T, hidden_dim)
        preds = self.head(gru_out).squeeze(-1)  # (1, T)
        return preds, hidden, gru_out


# ============================================================
# 4. TRAINING
# ============================================================

def train_epoch(model, dataset, optimizer, device):
    model.train()
    total_loss = 0
    valid_count = 0

    for item in dataset:
        features = item['features'].unsqueeze(0).to(device)  # (1, T, F)
        targets = item['targets'].unsqueeze(0).to(device)     # (1, T)
        mask = item['mask'].unsqueeze(0).to(device)            # (1, T)

        optimizer.zero_grad()
        preds, _, _ = model(features)

        # MSE loss only on valid (non-padded) timesteps
        loss = ((preds - targets) ** 2 * mask).sum() / mask.sum()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        valid_count += mask.sum().item()

    return total_loss / valid_count if valid_count > 0 else 0


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    all_preds = []
    all_targets = []

    for item in dataset:
        features = item['features'].unsqueeze(0).to(device)
        targets = item['targets'].numpy()
        mask = item['mask'].numpy().astype(bool)

        preds, _, _ = model(features)
        preds = preds.squeeze(0).cpu().numpy()

        all_preds.append(preds[mask])
        all_targets.append(targets[mask])

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    corr = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 1 else 0

    return {'mae': mae, 'rmse': rmse, 'corr': corr}


# ============================================================
# 5. LATENT CURVE EXTRACTION & PLOTTING
# ============================================================

@torch.no_grad()
def extract_latent_curves(model, dataset, device):
    """Extract latent skill estimates for all runs."""
    model.eval()
    curves = {}

    for item in dataset:
        features = item['features'].unsqueeze(0).to(device)
        preds, _, latent = model(features)

        curves[item['run_id']] = {
            'episodes': item['episodes'],
            'raw_scores': item['scores'].numpy(),
            'latent_skill': preds.squeeze(0).cpu().numpy(),
        }

    return curves


def plot_latent_curves(curves, output_path, num_samples=4):
    """Plot raw scores vs latent skill estimate for sample runs."""
    fig, axes = plt.subplots(min(num_samples, len(curves)), 1,
                              figsize=(14, 4 * min(num_samples, len(curves))))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    run_ids = list(curves.keys())[:num_samples]

    for ax, run_id in zip(axes, run_ids):
        c = curves[run_id]
        eps = c['episodes']

        # Plot raw scores (thin, transparent)
        ax.plot(eps, c['raw_scores'], alpha=0.3, linewidth=0.5,
                color='gray', label='Raw performance_score')

        # Plot latent skill estimate (bold)
        ax.plot(eps, c['latent_skill'], linewidth=2,
                color='blue', label='GRU latent skill estimate')

        ax.set_title(f'Run: {run_id}', fontsize=10)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


# ============================================================
# 6. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train GRU Latent Skill Tracker')
    parser.add_argument('--csv', type=str, required=True, help='Path to episode metrics CSV')
    parser.add_argument('--H', type=int, default=10, help='Future horizon for target averaging')
    parser.add_argument('--hidden_dim', type=int, default=64, help='GRU hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GRU layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--output_dir', type=str, default='./gru_output', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: H={args.H}, hidden={args.hidden_dim}, layers={args.num_layers}, lr={args.lr}")

    # --- Load data ---
    print("\nLoading data...")
    sequences, feature_cols = load_and_process(args.csv)
    print(f"Loaded {len(sequences)} runs, features: {feature_cols}")

    # --- Train/Val split by run_id ---
    run_ids = sorted(sequences.keys())
    np.random.seed(42)
    np.random.shuffle(run_ids)

    n_val = max(1, int(len(run_ids) * args.val_ratio))
    val_ids = set(run_ids[:n_val])
    train_ids = set(run_ids[n_val:])

    train_seqs = {k: v for k, v in sequences.items() if k in train_ids}
    val_seqs = {k: v for k, v in sequences.items() if k in val_ids}

    print(f"Train: {len(train_seqs)} runs, Val: {len(val_seqs)} runs")

    train_dataset = EpisodeSequenceDataset(train_seqs, args.H)
    val_dataset = EpisodeSequenceDataset(val_seqs, args.H)

    # --- Model ---
    input_dim = len(feature_cols)
    model = LatentGRU(input_dim, args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Input dim: {input_dim}, Hidden: {args.hidden_dim}, Layers: {args.num_layers}")

    # --- Training loop ---
    print("\n" + "="*60)
    print("Training...")
    print("="*60)

    best_val_loss = float('inf')
    best_epoch = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dataset, optimizer, device)
        val_metrics = evaluate(model, val_dataset, device)

        scheduler.step(val_metrics['rmse'])

        if val_metrics['rmse'] < best_val_loss:
            best_val_loss = val_metrics['rmse']
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / 'latent_gru_best.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.6f} | "
                  f"Val MAE: {val_metrics['mae']:.4f} | "
                  f"Val RMSE: {val_metrics['rmse']:.4f} | "
                  f"Val Corr: {val_metrics['corr']:.4f}")

    print(f"\nBest epoch: {best_epoch}, Best Val RMSE: {best_val_loss:.4f}")

    # --- Load best model & final evaluation ---
    model.load_state_dict(torch.load(output_dir / 'latent_gru_best.pt'))

    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    train_metrics = evaluate(model, train_dataset, device)
    val_metrics = evaluate(model, val_dataset, device)

    print(f"Train | MAE: {train_metrics['mae']:.4f} | RMSE: {train_metrics['rmse']:.4f} | Corr: {train_metrics['corr']:.4f}")
    print(f"Val   | MAE: {val_metrics['mae']:.4f} | RMSE: {val_metrics['rmse']:.4f} | Corr: {val_metrics['corr']:.4f}")

    # --- Extract & plot latent curves ---
    print("\nExtracting latent curves...")
    all_dataset = EpisodeSequenceDataset(sequences, args.H)
    curves = extract_latent_curves(model, all_dataset, device)

    plot_latent_curves(curves, output_dir / 'latent_skill_curves.png', num_samples=4)

    # --- Save latent curves to CSV ---
    rows = []
    for run_id, c in curves.items():
        for i, ep in enumerate(c['episodes']):
            rows.append({
                'run_id': run_id,
                'episode': ep,
                'raw_score': c['raw_scores'][i],
                'latent_skill': c['latent_skill'][i],
            })
    curves_df = pd.DataFrame(rows)
    curves_df.to_csv(output_dir / 'latent_skill_estimates.csv', index=False)
    print(f"Latent estimates saved: {output_dir / 'latent_skill_estimates.csv'}")

    # --- Save config ---
    config = vars(args)
    config['best_epoch'] = best_epoch
    config['train_metrics'] = train_metrics
    config['val_metrics'] = val_metrics
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print("\nDone!")


if __name__ == '__main__':
    main()