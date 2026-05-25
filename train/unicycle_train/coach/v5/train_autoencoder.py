# train/unicycle_train/coach/v5/train_autoencoder.py

"""
V5 Trajectory Autoencoder: Training Loop

Trains the TrajectoryAutoencoder on the V5 processed dataset
(train.npz / val.npz produced by build_dataset.py).

Key features:
  - Loads pre-built normalized dataset directly (no on-the-fly normalize).
  - DataLoader with shuffling for train, sequential for val.
  - AdamW optimizer with linear warmup + cosine decay scheduler.
  - Masked MSE loss (padding excluded).
  - Latent collapse monitoring: tracks std(z) per epoch. If z variance
    shrinks toward 0, the encoder is mapping everything to the same point
    and the framework is broken. This must be watched every epoch.
  - Checkpoint saving on best val loss.

Usage:
    python -m train.unicycle_train.coach.v5.train_autoencoder \
        --data-dir train/unicycle_train/coach/v5/dataset_processed \
        --output-dir train/unicycle_train/coach/v5/runs/ae_v5_<tag> \
        --batch-size 64 \
        --epochs 30 \
        --lr 1e-3
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

from train.unicycle_train.coach.v5.autoencoder_model import (
    TrajectoryAutoencoder,
    masked_mse_loss,
    count_parameters,
)


class TrajectoryDataset(Dataset):
    """Wraps the V5 processed npz into a torch Dataset.
    
    Loads the entire npz once (memory-mapped via np.load(..., mmap_mode='r')
    would be an option for huge datasets, but our train.npz is 20GB and
    your workstation has 125GB RAM, so we load directly for speed).
    """
    def __init__(self, npz_path):
        print(f"  Loading {npz_path}...")
        t0 = time.time()
        data = np.load(npz_path)
        self.features = data['features']  # (N, T_max, 21) float32
        self.masks = data['masks']        # (N, T_max) bool
        self.seeds = data['seeds']        # (N,) int32
        self.ep_ids = data['ep_ids']      # (N,) int32
        self.feature_mean = data['feature_mean']  # (21,)
        self.feature_std = data['feature_std']    # (21,)
        print(f"  Loaded {len(self.features)} episodes in {time.time()-t0:.1f}s")
        print(f"  features shape: {self.features.shape}, dtype {self.features.dtype}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Return torch tensors. Features already normalized in build_dataset.
        x = torch.from_numpy(self.features[idx]).float()    # (T_max, 21)
        m = torch.from_numpy(self.masks[idx])               # (T_max,) bool
        return x, m


def make_lr_schedule(optimizer, warmup_steps, total_steps):
    """Linear warmup followed by cosine decay to 10% of base lr.
    
    Warmup helps GRU training start stably (sudden large updates early
    can destabilize recurrent connections). Cosine decay improves final
    convergence quality.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        # Cosine from 1.0 down to 0.1
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def compute_latent_stats(model, loader, device, max_batches=20):
    """Sample a subset of batches and compute statistics of z.
    
    Monitors latent collapse: if std(z) shrinks toward 0, the encoder is
    producing the same z for every episode. This is a fatal framework
    failure mode and must be caught early.
    
    Returns:
        z_std_mean: scalar, average per-dimension std across z's
        z_norm_mean: scalar, mean L2 norm of z's
    """
    model.eval()
    z_all = []
    with torch.no_grad():
        for i, (x, m) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            m = m.to(device)
            z = model.encode(x, m)  # (B, latent_dim)
            z_all.append(z.cpu())
    
    if not z_all:
        return 0.0, 0.0
    
    z_cat = torch.cat(z_all, dim=0)  # (N_sampled, latent_dim)
    # Per-dim std across episodes — if any dim collapses, framework broken
    z_std_per_dim = z_cat.std(dim=0)  # (latent_dim,)
    z_std_mean = z_std_per_dim.mean().item()
    z_norm_mean = z_cat.norm(dim=-1).mean().item()
    
    model.train()
    return z_std_mean, z_norm_mean


def train_epoch(model, loader, optimizer, scheduler, device, log_interval=50):
    """Run one epoch over the training set."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    t_start = time.time()
    
    for batch_idx, (x, m) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        x_recon, z = model(x, m)
        loss = masked_mse_loss(x_recon, x, m)
        loss.backward()
        
        # Gradient clipping — GRU gradients can explode otherwise
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if batch_idx % log_interval == 0:
            elapsed = time.time() - t_start
            rate = (batch_idx + 1) / max(elapsed, 0.001)
            lr_now = scheduler.get_last_lr()[0]
            print(f"    batch {batch_idx:5d}/{len(loader)}  "
                  f"loss={loss.item():.4f}  lr={lr_now:.2e}  "
                  f"{rate:.1f} batches/s")
    
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def evaluate(model, loader, device):
    """Run full pass over val set. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, m in loader:
            x = x.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            x_recon, z = model(x, m)
            loss = masked_mse_loss(x_recon, x, m)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="V5 Autoencoder Training")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Dir containing train.npz and val.npz")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Run directory for checkpoints and logs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Model hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Data
    print(f"\nLoading datasets...")
    data_dir = Path(args.data_dir)
    train_ds = TrajectoryDataset(data_dir / "train.npz")
    val_ds = TrajectoryDataset(data_dir / "val.npz")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    
    print(f"\nTrain batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    
    # Model
    print(f"\nBuilding model...")
    model = TrajectoryAutoencoder(
        feature_dim=21,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        max_seq_len=1500,
        dropout=args.dropout,
    ).to(device)
    
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Encoder: {count_parameters(model.encoder):,}")
    print(f"  Decoder: {count_parameters(model.decoder):,}")
    
    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = make_lr_schedule(optimizer, args.warmup_steps, total_steps)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"  Starting training: {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Eval
        val_loss = evaluate(model, val_loader, device)
        
        # Latent statistics (sample 20 val batches)
        z_std, z_norm = compute_latent_stats(model, val_loader, device, max_batches=20)
        
        epoch_time = time.time() - epoch_t0
        
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Latent z:   std_mean={z_std:.4f}  norm_mean={z_norm:.4f}")
        print(f"  Epoch time: {epoch_time:.0f}s")
        
        # Collapse warning
        if z_std < 0.05:
            print(f"  [WARN] Latent z std is very low ({z_std:.4f}) — possible collapse!")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "z_std": z_std,
            "z_norm": z_norm,
            "epoch_time": epoch_time,
        })
        
        # Save history every epoch (cheap, useful if training crashes)
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": vars(args),
            }
            torch.save(ckpt, output_dir / "best_model.pt")
            print(f"  [BEST] Saved new best model (val_loss={val_loss:.4f})")
    
    # Also save final checkpoint
    final_ckpt = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": vars(args),
    }
    torch.save(final_ckpt, output_dir / "final_model.pt")
    
    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()