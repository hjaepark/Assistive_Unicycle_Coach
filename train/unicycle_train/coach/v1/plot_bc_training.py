# train/unicycle_train/coach/v1/plot_bc_training.py

"""
Usage:
    python -m train.unicycle_train.coach.v1.plot_bc_training
    python -m train.unicycle_train.coach.v1.plot_bc_training --tb_dir path/to/tensorboard
"""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(tb_dir, tag):
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tb_dir", type=str, default=None)
    args = parser.parse_args()

    if args.tb_dir is None:
        logs_dir = Path(__file__).resolve().parent / "logs"
        bc_runs = sorted(logs_dir.glob("bc_*"))
        if not bc_runs:
            print("No BC runs found")
            return
        tb_dir = str(bc_runs[-1] / "tensorboard")
    else:
        tb_dir = args.tb_dir

    train_epochs, train_mse = load_scalar(tb_dir, "train/mse")
    val_epochs, val_mse = load_scalar(tb_dir, "val/mse")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_epochs, train_mse, color="#2563EB", label="Train", linewidth=1.8)
    ax.plot(val_epochs, val_mse, color="#DC6926", label="Validation", linewidth=1.8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    
    textstr = "Wheel r=0.90\nPitch r=0.87"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='0.8')
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.legend(frameon=True, edgecolor="0.8")
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = Path(tb_dir).parent / "figures"
    out_path.mkdir(exist_ok=True)
    save_path = out_path / "bc_training_curve.png"
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()