# train/unicycle_train/coach/v1/eval_bc.py

"""
Coach V1 - BC Pretrained Coach Evaluation

Evaluates the BC pretrained actor by:
  1. Dataset-level: prediction accuracy on held-out data
  2. Environment-level: run novice student with/without BC coach
  3. Visualization: delta predictions vs actual deltas

Usage:
    python -m train.unicycle_train.coach.v1.eval_bc
    python -m train.unicycle_train.coach.v1.eval_bc --checkpoint path/to/best_bc_actor.pth
    python -m train.unicycle_train.coach.v1.eval_bc --n_episodes 50
"""

import argparse
import json
import numpy as np
from pathlib import Path

import torch
import gymnasium as gym
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

import envs.unicycle
from train.unicycle_train.coach.v0.coach_agent import CoachActor


def find_latest_checkpoint(logs_dir: Path) -> Path:
    """Find the most recent best_bc_actor.pth."""
    bc_runs = sorted(logs_dir.glob("bc_*"))
    if not bc_runs:
        raise FileNotFoundError(f"No BC runs found in {logs_dir}")
    checkpoint = bc_runs[-1] / "checkpoints" / "best_bc_actor.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint in {bc_runs[-1]}")
    return checkpoint


def find_latest_dataset(datasets_dir: Path) -> Path:
    """Find the most recent .npz dataset."""
    npz_files = sorted(datasets_dir.glob("bc_dataset_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No dataset found in {datasets_dir}")
    return npz_files[-1]


def load_bc_actor(checkpoint_path: str, device: torch.device) -> CoachActor:
    """Load BC pretrained actor."""
    model = CoachActor(obs_dim=22, action_dim=2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_delta(actor: CoachActor, coach_input: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict delta from coach input (22D)."""
    inp = torch.as_tensor(coach_input, dtype=torch.float32, device=device)
    if inp.dim() == 1:
        inp = inp.unsqueeze(0)
    mu, _ = actor(inp)
    return torch.tanh(mu).squeeze(0).cpu().numpy()


def eval_on_dataset(actor: CoachActor, dataset_path: Path, device: torch.device, output_dir: Path):
    """Evaluate prediction accuracy on dataset."""
    data = np.load(dataset_path)
    coach_inputs = torch.from_numpy(data["coach_inputs"]).to(device)
    deltas = data["deltas"]

    # Predict in batches
    predictions = []
    batch_size = 4096
    actor.eval()
    with torch.no_grad():
        for i in range(0, len(coach_inputs), batch_size):
            batch = coach_inputs[i:i + batch_size]
            mu, _ = actor(batch)
            pred = torch.tanh(mu).cpu().numpy()
            predictions.append(pred)
    predictions = np.concatenate(predictions, axis=0)

    # Clamp targets same as training
    targets = np.clip(deltas, -1.0, 1.0)

    # Metrics
    mse = np.mean((predictions - targets) ** 2)
    mse_wheel = np.mean((predictions[:, 0] - targets[:, 0]) ** 2)
    mse_pitch = np.mean((predictions[:, 1] - targets[:, 1]) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    # Correlation
    corr_wheel = np.corrcoef(predictions[:, 0], targets[:, 0])[0, 1]
    corr_pitch = np.corrcoef(predictions[:, 1], targets[:, 1])[0, 1]

    print("\n  [Dataset evaluation]")
    print(f"    MSE:          {mse:.6f} (wheel={mse_wheel:.6f}, pitch={mse_pitch:.6f})")
    print(f"    MAE:          {mae:.6f}")
    print(f"    Correlation:  wheel={corr_wheel:.4f}, pitch={corr_pitch:.4f}")

    return {"mse": mse, "mse_wheel": mse_wheel, "mse_pitch": mse_pitch,
            "mae": mae, "corr_wheel": corr_wheel, "corr_pitch": corr_pitch}

def eval_in_environment(
    actor: CoachActor,
    device: torch.device,
    n_episodes: int,
    output_dir: Path,
):
    """
    Run episodes with and without BC coach in the environment.
    Uses a fresh novice student (random policy) to test coach intervention.
    """
    env = gym.make("UnicyclePendulumPD-v0", reward_weights={
        "posture": 0.0,
        "pendulum": 0.0,
        "speed_reward_weight": 0.0,
        "alive_bonus": 0.0,
        "upright_gate_k": 0.0,
        "progress": 60.0,
        "goal_bonus": 50.0,
        "fall_penalty": -20.0,
    })

    results = {"no_coach": [], "bc_coach": []}

    for condition in ["no_coach", "bc_coach"]:
        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            ep_reward = 0.0
            ep_steps = 0
            max_x = 0.0

            while True:
                # Novice action: random within action space
                novice_action = env.action_space.sample()

                if condition == "bc_coach":
                    # Build coach input (22D)
                    coach_input = np.concatenate([obs, novice_action, [ep_reward / max(ep_steps, 1)]])
                    delta = predict_delta(actor, coach_input, device)
                    action = np.clip(novice_action + delta, -1.0, 1.0)
                else:
                    action = novice_action

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                max_x = max(max_x, info.get("x_position", 0.0))

                if terminated or truncated:
                    break

            results[condition].append({
                "reward": ep_reward,
                "steps": ep_steps,
                "max_x": max_x,
                "goal_reached": info.get("goal_reached", False),
                "is_fallen": info.get("is_fallen", False),
            })

    env.close()

    # Print summary
    print("\n  [Environment evaluation]")
    print(f"    Episodes per condition: {n_episodes}")
    print()

    for condition in ["no_coach", "bc_coach"]:
        r = results[condition]
        rewards = [e["reward"] for e in r]
        max_xs = [e["max_x"] for e in r]
        steps = [e["steps"] for e in r]
        successes = sum(1 for e in r if e["goal_reached"])
        falls = sum(1 for e in r if e["is_fallen"])

        label = "No coach (random)" if condition == "no_coach" else "BC coach + random"
        print(f"    {label}:")
        print(f"      Reward:    {np.mean(rewards):+.2f} +/- {np.std(rewards):.2f}")
        print(f"      Max X:     {np.mean(max_xs):.2f} +/- {np.std(max_xs):.2f}")
        print(f"      Steps:     {np.mean(steps):.0f} +/- {np.std(steps):.0f}")
        print(f"      Success:   {successes}/{n_episodes}")
        print(f"      Falls:     {falls}/{n_episodes}")
        print()

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (metric, label) in enumerate([("reward", "Episode Reward"), ("max_x", "Max X Position"), ("steps", "Episode Length")]):
        no_coach_vals = [e[metric] for e in results["no_coach"]]
        bc_coach_vals = [e[metric] for e in results["bc_coach"]]

        axes[i].boxplot([no_coach_vals, bc_coach_vals], tick_labels=["Novice student", "Novice student\n+ BC pre-trained coach"])
        axes[i].set_ylabel(label, fontsize=11)
        axes[i].set_title(label, fontsize=12)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "env_comparison.png", dpi=150)
    plt.close()
    print(f"    Saved: {output_dir / 'env_comparison.png'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC pretrained coach")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best_bc_actor.pth")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to .npz dataset")
    parser.add_argument("--n_episodes", type=int, default=30,
                        help="Episodes per condition for env evaluation")
    parser.add_argument("--skip_env", action="store_true",
                        help="Skip environment evaluation (dataset only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir / "logs"
    datasets_dir = script_dir / "datasets"

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Find checkpoint
    if args.checkpoint is None:
        checkpoint_path = find_latest_checkpoint(logs_dir)
    else:
        checkpoint_path = Path(args.checkpoint)

    # Find dataset
    if args.dataset is None:
        dataset_path = find_latest_dataset(datasets_dir)
    else:
        dataset_path = Path(args.dataset)

    # Output directory
    output_dir = checkpoint_path.parent.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    actor = load_bc_actor(str(checkpoint_path), device)

    print("=" * 70)
    print("Coach V1 - BC Evaluation")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Dataset:    {dataset_path.name}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")
    print("=" * 70)

    # 1. Dataset evaluation
    dataset_metrics = eval_on_dataset(actor, dataset_path, device, output_dir)

    # 2. Environment evaluation
    env_results = None
    if not args.skip_env:
        env_results = eval_in_environment(actor, device, args.n_episodes, output_dir)

    # Save all results
    all_results = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "dataset_metrics": dataset_metrics,
    }
    if env_results:
        for condition in env_results:
            rewards = [e["reward"] for e in env_results[condition]]
            max_xs = [e["max_x"] for e in env_results[condition]]
            successes = sum(1 for e in env_results[condition] if e["goal_reached"])
            all_results[f"env_{condition}"] = {
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "max_x_mean": float(np.mean(max_xs)),
                "success_count": successes,
                "n_episodes": args.n_episodes,
            }

    with open(output_dir / "evaluation_results.json", 'w') as f:
        all_results = json.loads(json.dumps(all_results, default=lambda x: float(x) if hasattr(x, 'item') else x))
        json.dump(all_results, f, indent=2)

    print(f"\n  Results saved: {output_dir / 'evaluation_results.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()