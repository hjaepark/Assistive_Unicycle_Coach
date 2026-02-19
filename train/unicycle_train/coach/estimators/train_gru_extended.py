"""
GRU Latent Skill Tracker - Extended Features
==============================================
Same pipeline as before but with additional input features:
  - performance_score
  - termination one-hot (fall, timeout, success)
  - T (episode length, normalized)
  - maximum_distance (normalized)
  - control_effort (normalized)
  - action_flip_rate (normalized)

Runs the full pipeline:
  1. Train on RL data
  2. Evaluate on human data (zero-shot)
  3. Fine-tune on human data (leave-one-out CV)
  4. Compare with baseline (4-feature model)

Usage:
    python train_gru_extended.py \
        --rl_csv <path_to_rl_csv> \
        --human_csv <path_to_human_csv> \
        --H 10 --epochs 50 --ft_epochs 30
"""

import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 1. MODEL
# ============================================================

class LatentGRU(nn.Module):
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
        gru_out, hidden = self.gru(x, hidden)
        preds = self.head(gru_out).squeeze(-1)
        return preds, hidden, gru_out


# ============================================================
# 2. DATA PROCESSING
# ============================================================

EXTENDED_FEATURE_COLS = [
    'performance_score',
    'term_fall', 'term_timeout', 'term_success',
    'T_norm', 'max_dist_norm', 'control_effort_norm', 'flip_rate_norm',
]

def normalize_column(series):
    """Min-max normalize a column to [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val < 1e-8:
        return series * 0.0
    return (series - min_val) / (max_val - min_val)


def load_and_process(csv_path, group_col='run_id'):
    """Load CSV, add extended features, group by run/participant."""
    df = pd.read_csv(csv_path)
    df = df.sort_values([group_col, 'episode']).reset_index(drop=True)

    # Termination one-hot
    df['term_fall'] = (df['termination'] == 'fall').astype(float)
    df['term_timeout'] = (df['termination'] == 'timeout').astype(float)
    df['term_success'] = (df['termination'] == 'success').astype(float)

    # Normalize additional features to [0, 1]
    df['T_norm'] = normalize_column(df['T'].astype(float))
    df['max_dist_norm'] = normalize_column(df['maximum_distance'])
    df['control_effort_norm'] = normalize_column(df['control_effort'])
    df['flip_rate_norm'] = normalize_column(df['action_flip_rate'])

    sequences = {}
    for gid, group in df.groupby(group_col):
        sequences[gid] = {
            'features': group[EXTENDED_FEATURE_COLS].values.astype(np.float32),
            'scores': group['performance_score'].values.astype(np.float32),
            'episodes': group['episode'].values,
            'terminations': group['termination'].values,
        }

    return sequences


def build_targets(scores, H):
    T = len(scores)
    targets = np.full(T, np.nan, dtype=np.float32)
    for k in range(T - H):
        targets[k] = np.mean(scores[k+1 : k+1+H])
    return targets


def prepare_items(sequences, H):
    """Convert sequences dict to list of training items."""
    items = {}
    for gid, seq in sequences.items():
        targets = build_targets(seq['scores'], H)
        mask = ~np.isnan(targets)
        items[gid] = {
            'id': gid,
            'features': torch.tensor(seq['features']),
            'targets': torch.tensor(np.nan_to_num(targets, nan=0.0)),
            'mask': torch.tensor(mask),
            'scores': torch.tensor(seq['scores']),
            'episodes': seq['episodes'],
        }
    return items


# ============================================================
# 3. TRAINING & EVALUATION
# ============================================================

def train_epoch(model, items, optimizer, device):
    model.train()
    total_loss = 0
    valid_count = 0

    for item in items:
        features = item['features'].unsqueeze(0).to(device)
        targets = item['targets'].unsqueeze(0).to(device)
        mask = item['mask'].unsqueeze(0).to(device)

        optimizer.zero_grad()
        preds, _, _ = model(features)
        loss = ((preds - targets) ** 2 * mask).sum() / mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        valid_count += mask.sum().item()

    return total_loss / valid_count if valid_count > 0 else 0


@torch.no_grad()
def evaluate_items(model, items, device):
    model.eval()
    all_preds, all_targets = [], []

    for item in items:
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
    return {'mae': float(mae), 'rmse': float(rmse), 'corr': float(corr)}


@torch.no_grad()
def evaluate_single(model, item, device):
    model.eval()
    features = item['features'].unsqueeze(0).to(device)
    targets = item['targets'].numpy()
    mask = item['mask'].numpy().astype(bool)

    preds, _, _ = model(features)
    preds = preds.squeeze(0).cpu().numpy()

    if mask.sum() > 0:
        valid_p, valid_t = preds[mask], targets[mask]
        mae = float(np.mean(np.abs(valid_p - valid_t)))
        rmse = float(np.sqrt(np.mean((valid_p - valid_t) ** 2)))
        corr = float(np.corrcoef(valid_p, valid_t)[0, 1]) if len(valid_p) > 1 else 0.0
    else:
        mae, rmse, corr = 0.0, 0.0, 0.0

    return {'mae': mae, 'rmse': rmse, 'corr': corr, 'preds': preds}


# ============================================================
# 4. FULL PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl_csv', type=str, required=True)
    parser.add_argument('--human_csv', type=str, required=True)
    parser.add_argument('--H', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ft_lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50, help='RL pre-training epochs')
    parser.add_argument('--ft_epochs', type=int, default=30, help='Human fine-tuning epochs')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--output_dir', type=str, default='./gru_extended_output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dim = len(EXTENDED_FEATURE_COLS)  # 8

    print(f"Device: {device}")
    print(f"Features ({input_dim}): {EXTENDED_FEATURE_COLS}")
    print(f"Config: H={args.H}, hidden={args.hidden_dim}, layers={args.num_layers}")

    # ================================================================
    # PHASE 1: Train on RL data
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Train on RL data")
    print("=" * 60)

    rl_seqs = load_and_process(args.rl_csv, group_col='run_id')
    rl_items = prepare_items(rl_seqs, args.H)
    print(f"Loaded {len(rl_items)} RL runs")

    # Train/val split
    run_ids = sorted(rl_items.keys())
    np.random.seed(42)
    np.random.shuffle(run_ids)
    n_val = max(1, int(len(run_ids) * args.val_ratio))
    val_ids = set(run_ids[:n_val])
    train_ids = set(run_ids[n_val:])

    train_list = [rl_items[k] for k in train_ids]
    val_list = [rl_items[k] for k in val_ids]
    print(f"Train: {len(train_list)}, Val: {len(val_list)}")

    model = LatentGRU(input_dim, args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_rmse = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_list, optimizer, device)
        val_metrics = evaluate_items(model, val_list, device)
        scheduler.step(val_metrics['rmse'])

        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            torch.save(model.state_dict(), output_dir / 'rl_pretrained.pt')

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train MSE: {train_loss:.6f} | "
                  f"Val MAE: {val_metrics['mae']:.4f} | "
                  f"Val RMSE: {val_metrics['rmse']:.4f} | "
                  f"Val Corr: {val_metrics['corr']:.4f}")

    # Load best
    model.load_state_dict(torch.load(output_dir / 'rl_pretrained.pt'))
    rl_train_metrics = evaluate_items(model, train_list, device)
    rl_val_metrics = evaluate_items(model, val_list, device)
    print(f"\nRL Results:")
    print(f"  Train | MAE: {rl_train_metrics['mae']:.4f} | Corr: {rl_train_metrics['corr']:.4f}")
    print(f"  Val   | MAE: {rl_val_metrics['mae']:.4f} | Corr: {rl_val_metrics['corr']:.4f}")

    # ================================================================
    # PHASE 2: Zero-shot evaluation on human data
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Zero-shot on Human data")
    print("=" * 60)

    human_seqs = load_and_process(args.human_csv, group_col='participant_id')
    human_items = prepare_items(human_seqs, args.H)
    pids = sorted(human_items.keys())
    print(f"Loaded {len(human_items)} participants")

    zero_shot_results = {}
    for pid in pids:
        metrics = evaluate_single(model, human_items[pid], device)
        zero_shot_results[pid] = metrics
        print(f"  {pid} | MAE: {metrics['mae']:.4f} | Corr: {metrics['corr']:.4f}")

    avg_zs_corr = np.mean([r['corr'] for r in zero_shot_results.values()])
    print(f"  Average Corr: {avg_zs_corr:.4f}")

    # ================================================================
    # PHASE 3: Fine-tune with leave-one-out CV
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Human Fine-tune (Leave-One-Out CV)")
    print("=" * 60)

    ft_results = {}
    for test_pid in pids:
        print(f"\n--- Hold out {test_pid} ---")

        # Fresh copy of RL-pretrained model
        ft_model = LatentGRU(input_dim, args.hidden_dim, args.num_layers).to(device)
        ft_model.load_state_dict(torch.load(output_dir / 'rl_pretrained.pt'))
        ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=args.ft_lr)

        train_items = [human_items[pid] for pid in pids if pid != test_pid]
        best_state = None
        best_loss = float('inf')

        for epoch in range(1, args.ft_epochs + 1):
            loss = train_epoch(ft_model, train_items, ft_optimizer, device)
            if loss < best_loss:
                best_loss = loss
                best_state = copy.deepcopy(ft_model.state_dict())
            if epoch % 10 == 0:
                test_m = evaluate_single(ft_model, human_items[test_pid], device)
                print(f"  Epoch {epoch:3d} | Train MSE: {loss:.6f} | Test Corr: {test_m['corr']:.4f}")

        ft_model.load_state_dict(best_state)
        final_m = evaluate_single(ft_model, human_items[test_pid], device)
        ft_results[test_pid] = final_m
        print(f"  {test_pid} Final | MAE: {final_m['mae']:.4f} | Corr: {final_m['corr']:.4f}")

    avg_ft_corr = np.mean([r['corr'] for r in ft_results.values()])

    # ================================================================
    # PHASE 4: Summary & Comparison
    # ================================================================
    print("\n" + "=" * 60)
    print("FULL COMPARISON: Extended Features (8) vs Baseline (4)")
    print("=" * 60)
    print(f"\n{'PID':<6} {'Zero-shot':>10} {'Fine-tuned':>11}")
    print("-" * 30)
    for pid in pids:
        zs = zero_shot_results[pid]['corr']
        ft = ft_results[pid]['corr']
        print(f"{pid:<6} {zs:>10.4f} {ft:>11.4f}")
    print("-" * 30)
    print(f"{'AVG':<6} {avg_zs_corr:>10.4f} {avg_ft_corr:>11.4f}")

    # ================================================================
    # PHASE 5: Train final model on all human data & save
    # ================================================================
    print("\n" + "=" * 60)
    print("Training final model on ALL human data...")
    print("=" * 60)

    final_model = LatentGRU(input_dim, args.hidden_dim, args.num_layers).to(device)
    final_model.load_state_dict(torch.load(output_dir / 'rl_pretrained.pt'))
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.ft_lr)

    all_human_items = [human_items[pid] for pid in pids]
    for epoch in range(1, args.ft_epochs + 1):
        loss = train_epoch(final_model, all_human_items, final_optimizer, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train MSE: {loss:.6f}")

    torch.save(final_model.state_dict(), output_dir / 'latent_gru_extended_finetuned.pt')

    # ================================================================
    # PHASE 6: Plots
    # ================================================================

    # Plot per-participant comparison
    n = len(pids)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, pid in zip(axes, pids):
        eps = human_items[pid]['episodes']
        raw = human_items[pid]['scores'].numpy()

        ax.plot(eps, raw, alpha=0.3, linewidth=0.5, color='gray', label='Raw score')
        ax.plot(eps, zero_shot_results[pid]['preds'], linewidth=1.5,
                color='red', alpha=0.6, linestyle='--',
                label=f"Zero-shot (corr={zero_shot_results[pid]['corr']:.3f})")
        ax.plot(eps, ft_results[pid]['preds'], linewidth=2, color='blue',
                label=f"Fine-tuned (corr={ft_results[pid]['corr']:.3f})")

        ax.set_title(f"{pid} | Zero-shot: {zero_shot_results[pid]['corr']:.3f} → Fine-tuned: {ft_results[pid]['corr']:.3f}")
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'extended_comparison_curves.png', dpi=150)
    plt.close()

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(pids))
    width = 0.35

    zs_corrs = [zero_shot_results[pid]['corr'] for pid in pids]
    ft_corrs = [ft_results[pid]['corr'] for pid in pids]

    ax.bar(x - width/2, zs_corrs, width, label='Zero-shot (RL only)', color='red', alpha=0.6)
    ax.bar(x + width/2, ft_corrs, width, label='Fine-tuned', color='blue', alpha=0.6)
    ax.axhline(y=np.mean(zs_corrs), color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(ft_corrs), color='blue', linestyle='--', alpha=0.5)

    ax.set_xlabel('Participant')
    ax.set_ylabel('Correlation')
    ax.set_title(f'Extended Features ({input_dim}): Zero-shot vs Fine-tuned')
    ax.set_xticks(x)
    ax.set_xticklabels(pids)
    ax.legend()
    ax.set_ylim(-0.1, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'extended_bar_comparison.png', dpi=150)
    plt.close()

    # ================================================================
    # Save summary
    # ================================================================
    summary = {
        'features': EXTENDED_FEATURE_COLS,
        'input_dim': input_dim,
        'config': vars(args),
        'rl_train': rl_train_metrics,
        'rl_val': rl_val_metrics,
        'human_zero_shot_avg_corr': float(avg_zs_corr),
        'human_finetuned_avg_corr': float(avg_ft_corr),
        'per_participant': {
            pid: {
                'zero_shot_corr': float(zero_shot_results[pid]['corr']),
                'finetuned_corr': float(ft_results[pid]['corr']),
            } for pid in pids
        },
    }
    with open(output_dir / 'extended_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()