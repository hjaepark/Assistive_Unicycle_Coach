"""
Fine-tune GRU Latent Skill Tracker on Human Data
==================================================
Loads RL-pretrained GRU and fine-tunes on human participant data.
Uses leave-one-out cross-validation (train on 9, test on 1) to
evaluate generalization since we only have 10 participants.

Usage:
    python finetune_gru_human.py \
        --model_path ./gru_output/latent_gru_best.pt \
        --human_csv <path_to_human_csv> \
        --H 10 \
        --epochs 30 \
        --output_dir ./gru_human_finetune
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
# 1. MODEL (same architecture)
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

def load_human_data(csv_path):
    """Load human CSV and group by participant_id."""
    df = pd.read_csv(csv_path)
    df = df.sort_values(['participant_id', 'episode']).reset_index(drop=True)

    df['term_fall'] = (df['termination'] == 'fall').astype(float)
    df['term_timeout'] = (df['termination'] == 'timeout').astype(float)
    df['term_success'] = (df['termination'] == 'success').astype(float)

    feature_cols = ['performance_score', 'term_fall', 'term_timeout', 'term_success']

    participants = {}
    for pid, group in df.groupby('participant_id'):
        participants[pid] = {
            'features': torch.tensor(group[feature_cols].values.astype(np.float32)),
            'scores': group['performance_score'].values.astype(np.float32),
            'episodes': group['episode'].values,
            'terminations': group['termination'].values,
        }

    return participants, feature_cols


def build_targets(scores, H):
    T = len(scores)
    targets = np.full(T, np.nan, dtype=np.float32)
    for k in range(T - H):
        targets[k] = np.mean(scores[k+1 : k+1+H])
    return targets


def prepare_item(data, H):
    """Prepare a single participant's data for training/eval."""
    targets = build_targets(data['scores'], H)
    mask = ~np.isnan(targets)
    targets_clean = np.nan_to_num(targets, nan=0.0)
    return {
        'features': data['features'],
        'targets': torch.tensor(targets_clean),
        'mask': torch.tensor(mask),
        'scores': torch.tensor(data['scores']),
        'episodes': data['episodes'],
    }


# ============================================================
# 3. TRAINING & EVALUATION
# ============================================================

def train_epoch(model, train_items, optimizer, device):
    model.train()
    total_loss = 0
    valid_count = 0

    for item in train_items:
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
def evaluate_item(model, item, device):
    """Evaluate on a single participant."""
    model.eval()
    features = item['features'].unsqueeze(0).to(device)
    targets = item['targets'].numpy()
    mask = item['mask'].numpy().astype(bool)

    preds, _, _ = model(features)
    preds = preds.squeeze(0).cpu().numpy()

    if mask.sum() > 0:
        valid_p = preds[mask]
        valid_t = targets[mask]
        mae = np.mean(np.abs(valid_p - valid_t))
        rmse = np.sqrt(np.mean((valid_p - valid_t) ** 2))
        corr = np.corrcoef(valid_p, valid_t)[0, 1] if len(valid_p) > 1 else 0
    else:
        mae, rmse, corr = np.nan, np.nan, np.nan

    return {'mae': mae, 'rmse': rmse, 'corr': corr, 'preds': preds}


# ============================================================
# 4. LEAVE-ONE-OUT CROSS-VALIDATION
# ============================================================

def leave_one_out_cv(pretrained_path, participants, H, args, device):
    """
    For each participant:
      1. Load fresh copy of RL-pretrained model
      2. Fine-tune on other 9 participants
      3. Evaluate on held-out participant
    """
    pids = sorted(participants.keys())
    all_items = {pid: prepare_item(participants[pid], H) for pid in pids}

    cv_results = {}

    for test_pid in pids:
        print(f"\n--- Fold: hold out {test_pid} ---")

        # Load fresh pretrained weights
        input_dim = 4
        model = LatentGRU(input_dim, args.hidden_dim, args.num_layers).to(device)
        model.load_state_dict(torch.load(pretrained_path, map_location=device))

        # Split
        train_items = [all_items[pid] for pid in pids if pid != test_pid]
        test_item = all_items[test_pid]

        # Fine-tune with lower learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr)

        best_model_state = None
        best_train_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_items, optimizer, device)
            if loss < best_train_loss:
                best_train_loss = loss
                best_model_state = copy.deepcopy(model.state_dict())

            if epoch % 10 == 0:
                test_metrics = evaluate_item(model, test_item, device)
                print(f"  Epoch {epoch:3d} | Train MSE: {loss:.6f} | "
                      f"Test Corr: {test_metrics['corr']:.4f}")

        # Load best and evaluate
        model.load_state_dict(best_model_state)
        final_metrics = evaluate_item(model, test_item, device)

        cv_results[test_pid] = {
            'metrics': final_metrics,
            'preds': final_metrics['preds'],
            'raw_scores': all_items[test_pid]['scores'].numpy(),
            'episodes': participants[test_pid]['episodes'],
        }

        print(f"  {test_pid} Final | MAE: {final_metrics['mae']:.4f} | "
              f"RMSE: {final_metrics['rmse']:.4f} | Corr: {final_metrics['corr']:.4f}")

    return cv_results


# ============================================================
# 5. ALSO TRAIN ON ALL HUMAN DATA (final model)
# ============================================================

def train_final_model(pretrained_path, participants, H, args, device):
    """Fine-tune on ALL human participants for the final deployed model."""
    pids = sorted(participants.keys())
    all_items = [prepare_item(participants[pid], H) for pid in pids]

    input_dim = 4
    model = LatentGRU(input_dim, args.hidden_dim, args.num_layers).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr)

    print("\n" + "=" * 60)
    print("Training final model on ALL human data...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, all_items, optimizer, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train MSE: {loss:.6f}")

    return model


# ============================================================
# 6. PLOTTING
# ============================================================

def plot_cv_results(cv_results, before_results, output_dir):
    """Plot before vs after fine-tuning for each participant."""
    n = len(cv_results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, (pid, r) in zip(axes, sorted(cv_results.items())):
        eps = r['episodes']

        # Raw scores
        ax.plot(eps, r['raw_scores'], alpha=0.3, linewidth=0.5,
                color='gray', label='Raw performance_score')

        # Before fine-tune (if available)
        if pid in before_results:
            ax.plot(eps, before_results[pid]['preds'], linewidth=1.5,
                    color='red', alpha=0.5, linestyle='--',
                    label=f"Before FT (corr={before_results[pid]['metrics']['corr']:.3f})")

        # After fine-tune
        m = r['metrics']
        ax.plot(eps, r['preds'], linewidth=2,
                color='blue',
                label=f"After FT (corr={m['corr']:.3f})")

        ax.set_title(f'{pid} | Before→After Corr: '
                     f'{before_results.get(pid, {}).get("metrics", {}).get("corr", 0):.3f} → {m["corr"]:.3f}',
                     fontsize=10)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out_path = output_dir / 'finetune_comparison.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved: {out_path}")


def plot_bar_comparison(cv_results, before_results, output_dir):
    """Bar chart comparing corr before vs after fine-tuning."""
    pids = sorted(cv_results.keys())
    before_corrs = [before_results.get(pid, {}).get('metrics', {}).get('corr', 0) for pid in pids]
    after_corrs = [cv_results[pid]['metrics']['corr'] for pid in pids]

    x = np.arange(len(pids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, before_corrs, width, label='Before Fine-tune (RL only)', color='red', alpha=0.6)
    ax.bar(x + width/2, after_corrs, width, label='After Fine-tune', color='blue', alpha=0.6)

    ax.set_xlabel('Participant')
    ax.set_ylabel('Correlation')
    ax.set_title('Skill Estimator: Before vs After Human Fine-tuning')
    ax.set_xticks(x)
    ax.set_xticklabels(pids)
    ax.legend()
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Add average lines
    avg_before = np.mean(before_corrs)
    avg_after = np.mean(after_corrs)
    ax.axhline(y=avg_before, color='red', linestyle='--', alpha=0.5,
               label=f'Avg before: {avg_before:.3f}')
    ax.axhline(y=avg_after, color='blue', linestyle='--', alpha=0.5,
               label=f'Avg after: {avg_after:.3f}')
    ax.legend()

    out_path = output_dir / 'finetune_bar_comparison.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Bar comparison saved: {out_path}")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GRU on Human Data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to RL-pretrained latent_gru_best.pt')
    parser.add_argument('--human_csv', type=str, required=True,
                        help='Path to human training CSV')
    parser.add_argument('--H', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--ft_lr', type=float, default=1e-4,
                        help='Fine-tuning learning rate (lower than pre-training)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='./gru_human_finetune')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: H={args.H}, ft_lr={args.ft_lr}, epochs={args.epochs}")

    # --- Load human data ---
    participants, feature_cols = load_human_data(args.human_csv)
    print(f"Loaded {len(participants)} participants")

    # --- Get "before" results (RL-only model, no fine-tune) ---
    print("\n" + "=" * 60)
    print("Baseline: RL-pretrained model on human data (no fine-tune)")
    print("=" * 60)

    input_dim = 4
    baseline_model = LatentGRU(input_dim, args.hidden_dim, args.num_layers).to(device)
    baseline_model.load_state_dict(torch.load(args.model_path, map_location=device))

    before_results = {}
    for pid in sorted(participants.keys()):
        item = prepare_item(participants[pid], args.H)
        metrics = evaluate_item(baseline_model, item, device)
        before_results[pid] = {
            'metrics': metrics,
            'preds': metrics['preds'],
        }
        print(f"  {pid} | Corr: {metrics['corr']:.4f}")
    print(f"  Average Corr: {np.mean([r['metrics']['corr'] for r in before_results.values()]):.4f}")

    # --- Leave-one-out CV ---
    print("\n" + "=" * 60)
    print("Leave-One-Out Cross-Validation (fine-tune on 9, test on 1)")
    print("=" * 60)

    cv_results = leave_one_out_cv(args.model_path, participants, args.H, args, device)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY: Before vs After Fine-tuning")
    print("=" * 60)
    print(f"{'PID':<6} {'Before Corr':>12} {'After Corr':>12} {'Change':>10}")
    print("-" * 42)

    for pid in sorted(cv_results.keys()):
        before_c = before_results[pid]['metrics']['corr']
        after_c = cv_results[pid]['metrics']['corr']
        change = after_c - before_c
        arrow = "↑" if change > 0 else "↓"
        print(f"{pid:<6} {before_c:>12.4f} {after_c:>12.4f} {arrow} {abs(change):>8.4f}")

    avg_before = np.mean([before_results[pid]['metrics']['corr'] for pid in cv_results])
    avg_after = np.mean([cv_results[pid]['metrics']['corr'] for pid in cv_results])
    print("-" * 42)
    print(f"{'AVG':<6} {avg_before:>12.4f} {avg_after:>12.4f} {'↑' if avg_after > avg_before else '↓'} {abs(avg_after - avg_before):>8.4f}")

    # --- Train final model on all human data ---
    final_model = train_final_model(args.model_path, participants, args.H, args, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save final model
    torch.save(final_model.state_dict(), output_dir / 'latent_gru_human_finetuned.pt')
    print(f"\nFinal model saved: {output_dir / 'latent_gru_human_finetuned.pt'}")

    # --- Plots ---
    plot_cv_results(cv_results, before_results, output_dir)
    plot_bar_comparison(cv_results, before_results, output_dir)

    # --- Save results ---
    summary = {
        'config': vars(args),
        'before_avg_corr': float(avg_before),
        'after_avg_corr': float(avg_after),
        'per_participant': {},
    }
    for pid in sorted(cv_results.keys()):
        summary['per_participant'][pid] = {
            'before': {k: float(v) if not isinstance(v, np.ndarray) else None
                       for k, v in before_results[pid]['metrics'].items()},
            'after': {k: float(v) if not isinstance(v, np.ndarray) else None
                      for k, v in cv_results[pid]['metrics'].items()},
        }

    with open(output_dir / 'finetune_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved: {output_dir / 'finetune_summary.json'}")

    print("\nDone!")


if __name__ == '__main__':
    main()