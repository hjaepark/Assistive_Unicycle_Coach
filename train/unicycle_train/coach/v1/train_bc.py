# train/unicycle_train/coach/v1/train_bc.py

"""
Coach V1 - Behavior Cloning Pretraining

Trains the coach actor to predict action deltas (expert - novice)
using the V0 CoachActor architecture (256x256 MLP, tanh output).
The trained weights can be directly loaded into V0 CoachAgent.

Usage:
    python -m train.unicycle_train.coach.v1.train_bc
    python -m train.unicycle_train.coach.v1.train_bc --dataset path/to/bc_dataset.npz
    python -m train.unicycle_train.coach.v1.train_bc --epochs 200 --lr 1e-4
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Reuse V0 architecture for weight compatibility
from train.unicycle_train.coach.v0.coach_agent import CoachActor


class BCDataset(Dataset):
    """PyTorch dataset for BC training from .npz file."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.inputs = torch.from_numpy(data["coach_inputs"])   # (N, 22)
        self.targets = torch.from_numpy(data["deltas"])         # (N, 2)

        # Clamp targets to [-1, 1] since CoachActor outputs tanh
        self.targets = torch.clamp(self.targets, -1.0, 1.0)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def find_latest_dataset(datasets_dir: Path) -> Path:
    """Find the most recent .npz dataset file."""
    npz_files = sorted(datasets_dir.glob("bc_dataset_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No dataset found in {datasets_dir}")
    return npz_files[-1]


def evaluate(model, dataloader, device):
    """Compute validation loss and per-action-dim MSE."""
    model.eval()
    total_loss = 0.0
    total_dim0 = 0.0
    total_dim1 = 0.0
    n = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            mu, _ = model(inputs)
            pred = torch.tanh(mu)

            mse = (pred - targets).pow(2)
            total_loss += mse.sum().item()
            total_dim0 += mse[:, 0].sum().item()
            total_dim1 += mse[:, 1].sum().item()
            n += len(inputs)

    model.train()
    return {
        "mse": total_loss / n / 2,
        "mse_wheel": total_dim0 / n,
        "mse_pitch": total_dim1 / n,
    }


def main():
    parser = argparse.ArgumentParser(description="BC pretraining for Coach V1")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to .npz dataset (default: latest in datasets/)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of data used for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=20,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).resolve().parent
    datasets_dir = script_dir / "datasets"
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    if args.dataset is None:
        dataset_path = find_latest_dataset(datasets_dir)
    else:
        dataset_path = Path(args.dataset)

    dataset = BCDataset(dataset_path)
    n_total = len(dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model — V0 CoachActor architecture (obs_dim=22, action_dim=2)
    model = CoachActor(obs_dim=22, action_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = logs_dir / f"bc_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    # Save config
    config = {
        "dataset": str(dataset_path),
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "device": str(device),
        "architecture": "V0 CoachActor (256x256 MLP, tanh-normal)",
        "obs_dim": 22,
        "action_dim": 2,
    }
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("Coach V1 - BC Pretraining")
    print("=" * 70)
    print(f"  Dataset:    {dataset_path.name}")
    print(f"  Samples:    {n_train:,} train / {n_val:,} val")
    print(f"  Model:      CoachActor (22 → 256 → 256 → 2, tanh)")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {device}")
    print(f"  Run dir:    {run_dir}")
    print("=" * 70)

    # Training loop
    best_val_mse = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward: CoachActor returns (mu, std)
            # BC loss: MSE between tanh(mu) and target delta
            mu, _ = model(inputs)
            pred = torch.tanh(mu)
            loss = nn.functional.mse_loss(pred, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        train_mse = epoch_loss / n_batches
        val_metrics = evaluate(model, val_loader, device)
        val_mse = val_metrics["mse"]

        # Tensorboard
        writer.add_scalar("train/mse", train_mse, epoch)
        writer.add_scalar("val/mse", val_mse, epoch)
        writer.add_scalar("val/mse_wheel", val_metrics["mse_wheel"], epoch)
        writer.add_scalar("val/mse_pitch", val_metrics["mse_pitch"], epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # Save best
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), checkpoint_dir / "best_bc_actor.pth")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"bc_actor_epoch_{epoch:04d}.pth")

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:4d}/{args.epochs} | "
                  f"train_mse={train_mse:.6f} | "
                  f"val_mse={val_mse:.6f} (wheel={val_metrics['mse_wheel']:.6f}, pitch={val_metrics['mse_pitch']:.6f}) | "
                  f"best={best_val_mse:.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"{elapsed:.0f}s")

    # Save final
    torch.save(model.state_dict(), checkpoint_dir / "final_bc_actor.pth")
    writer.close()

    total_time = time.time() - start_time

    # Save results summary
    results = {
        "best_val_mse": best_val_mse,
        "final_train_mse": train_mse,
        "final_val_mse": val_mse,
        "total_time_seconds": round(total_time, 1),
        "best_checkpoint": str(checkpoint_dir / "best_bc_actor.pth"),
    }
    with open(run_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print("BC PRETRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best val MSE:  {best_val_mse:.6f}")
    print(f"  Total time:    {total_time:.1f}s")
    print(f"  Best model:    {checkpoint_dir / 'best_bc_actor.pth'}")
    print()
    print("  To load into V0 CoachAgent:")
    print("    coach = CoachAgent(obs_dim=22, action_dim=2)")
    print(f"    coach.actor.load_state_dict(torch.load('{checkpoint_dir / 'best_bc_actor.pth'}'))")
    print("=" * 70)


if __name__ == "__main__":
    main()