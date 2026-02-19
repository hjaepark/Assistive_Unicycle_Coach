"""
Evaluate trained GRU Latent Skill Tracker on Human Data
=========================================================
Loads a pre-trained GRU model (trained on RL data) and runs inference
on human participant data to check generalization.

Usage:
    python evaluate_gru_human.py \
        --model_path ./gru_output/latent_gru_best.pt \
        --human_csv <path_to_human_csv> \
        --H 10 \
        --output_dir ./gru_human_eval
"""

import argparse
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
# 1. MODEL (same architecture as training script)
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

    # One-hot encode termination (same as training)
    df['term_fall'] = (df['termination'] == 'fall').astype(float)
    df['term_timeout'] = (df['termination'] == 'timeout').astype(float)
    df['term_success'] = (df['termination'] == 'success').astype(float)

    feature_cols = ['performance_score', 'term_fall', 'term_timeout', 'term_success']

    participants = {}
    for pid, group in df.groupby('participant_id'):
        participants[pid] = {
            'features': group[feature_cols].values.astype(np.float32),
            'scores': group['performance_score'].values.astype(np.float32),
            'episodes': group['episode'].values,
            'terminations': group['termination'].values,
            'T': group['T'].values,
        }

    return participants, feature_cols


def build_targets(scores, H):
    """Build future-H-average targets."""
    T = len(scores)
    targets = np.full(T, np.nan, dtype=np.float32)
    for k in range(T - H):
        targets[k] = np.mean(scores[k+1 : k+1+H])
    return targets


# ============================================================
# 3. INFERENCE & METRICS
# ============================================================

@torch.no_grad()
def run_inference(model, participants, H, device):
    """Run GRU inference on all participants."""
    model.eval()
    results = {}

    for pid, data in participants.items():
        features = torch.tensor(data['features']).unsqueeze(0).to(device)  # (1, T, F)
        preds, _, latent = model(features)
        preds = preds.squeeze(0).cpu().numpy()

        targets = build_targets(data['scores'], H)
        mask = ~np.isnan(targets)

        # Compute metrics where targets are valid
        if mask.sum() > 0:
            valid_preds = preds[mask]
            valid_targets = targets[mask]
            mae = np.mean(np.abs(valid_preds - valid_targets))
            rmse = np.sqrt(np.mean((valid_preds - valid_targets) ** 2))
            corr = np.corrcoef(valid_preds, valid_targets)[0, 1] if len(valid_preds) > 1 else 0
        else:
            mae, rmse, corr = np.nan, np.nan, np.nan

        # Compute success rate (rolling window for comparison)
        window = 20
        success_rate = np.array([
            np.mean(data['terminations'][max(0, i-window):i+1] == 'success')
            for i in range(len(data['terminations']))
        ])

        results[pid] = {
            'episodes': data['episodes'],
            'raw_scores': data['scores'],
            'latent_skill': preds,
            'targets': targets,
            'mask': mask,
            'success_rate': success_rate,
            'T': data['T'],
            'metrics': {'mae': mae, 'rmse': rmse, 'corr': corr},
        }

    return results


# ============================================================
# 4. PLOTTING
# ============================================================

def plot_all_participants(results, output_dir):
    """Plot latent skill curves for all human participants."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, (pid, r) in zip(axes, sorted(results.items())):
        eps = r['episodes']

        # Raw scores (thin, transparent)
        ax.plot(eps, r['raw_scores'], alpha=0.3, linewidth=0.5,
                color='gray', label='Raw performance_score')

        # GRU latent skill (bold blue)
        ax.plot(eps, r['latent_skill'], linewidth=2,
                color='blue', label='GRU latent skill')

        # Success rate (green dashed, for comparison)
        ax.plot(eps, r['success_rate'], linewidth=1.5,
                color='green', alpha=0.6, linestyle='--',
                label=f'Success rate (window=20)')

        m = r['metrics']
        ax.set_title(f'{pid} | MAE: {m["mae"]:.4f} | RMSE: {m["rmse"]:.4f} | Corr: {m["corr"]:.4f}',
                     fontsize=10)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out_path = output_dir / 'human_latent_skill_curves.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path}")


def plot_summary(results, output_dir):
    """Plot all participants' latent skills overlaid on one chart."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (pid, r), color in zip(sorted(results.items()), colors):
        eps = r['episodes']
        ax.plot(eps, r['latent_skill'], linewidth=1.5, color=color, label=pid)

    ax.set_title('All Participants - GRU Latent Skill Estimate', fontsize=12)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Latent Skill')
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-0.05, 1.05)

    out_path = output_dir / 'human_latent_skill_summary.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Summary plot saved: {out_path}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate GRU on Human Data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained latent_gru_best.pt')
    parser.add_argument('--human_csv', type=str, required=True,
                        help='Path to human training CSV')
    parser.add_argument('--H', type=int, default=10,
                        help='Future horizon (must match training H)')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='./gru_human_eval')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load model ---
    input_dim = 4  # performance_score + 3 termination one-hot
    model = LatentGRU(input_dim, args.hidden_dim, args.num_layers).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded: {args.model_path}")

    # --- Load human data ---
    participants, feature_cols = load_human_data(args.human_csv)
    print(f"Loaded {len(participants)} participants")
    for pid, data in sorted(participants.items()):
        print(f"  {pid}: {len(data['scores'])} episodes")

    # --- Run inference ---
    print("\nRunning inference...")
    results = run_inference(model, participants, args.H, device)

    # --- Print metrics ---
    print("\n" + "=" * 60)
    print("Human Data Evaluation (model trained on RL data)")
    print("=" * 60)

    all_mae, all_corr = [], []
    for pid, r in sorted(results.items()):
        m = r['metrics']
        print(f"  {pid} | MAE: {m['mae']:.4f} | RMSE: {m['rmse']:.4f} | Corr: {m['corr']:.4f}")
        if not np.isnan(m['mae']):
            all_mae.append(m['mae'])
            all_corr.append(m['corr'])

    print(f"\n  Average MAE:  {np.mean(all_mae):.4f}")
    print(f"  Average Corr: {np.mean(all_corr):.4f}")

    # --- Plot ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_all_participants(results, output_dir)
    plot_summary(results, output_dir)

    # --- Save latent estimates CSV ---
    rows = []
    for pid, r in results.items():
        for i, ep in enumerate(r['episodes']):
            rows.append({
                'participant_id': pid,
                'episode': ep,
                'raw_score': r['raw_scores'][i],
                'latent_skill': r['latent_skill'][i],
                'success_rate_20': r['success_rate'][i],
            })
    out_df = pd.DataFrame(rows)
    csv_path = output_dir / 'human_latent_skill_estimates.csv'
    out_df.to_csv(csv_path, index=False)
    print(f"\nEstimates saved: {csv_path}")

    # --- Save summary ---
    summary = {
        'model_path': args.model_path,
        'human_csv': args.human_csv,
        'H': args.H,
        'num_participants': len(participants),
        'per_participant': {pid: r['metrics'] for pid, r in results.items()},
        'avg_mae': float(np.mean(all_mae)),
        'avg_corr': float(np.mean(all_corr)),
    }
    with open(output_dir / 'human_eval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nDone!")


if __name__ == '__main__':
    main()