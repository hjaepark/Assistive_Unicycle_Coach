# train/student/eval_student.py

# To evaluate baseline data, use: python -m train.student.eval_student train/student/data/baseline/seed*

import re
import argparse
from pathlib import Path
import numpy as np
import gymnasium as gym
import envs.unicycle
from stable_baselines3 import SAC
from train.student.train_student import REWARD

EVAL_OFFSET = 10000


def parse_seed(run_dir):
    m = re.search(r"seed(\d+)", run_dir.name)
    if m is None:
        raise ValueError(f"cannot parse seed from: {run_dir.name}")
    return int(m.group(1))


def make_env(seed):
    env = gym.make("UnicyclePendulumPD-v0", reward_weights=REWARD, frame_skip=1)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def rollout(model, env, seed):
    obs, info = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    length = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        length += 1
        done = terminated or truncated
    return {
        "return": ep_return,
        "goal_reached": bool(info["goal_reached"]),
        "is_fallen": bool(info["is_fallen"]),
        "length": length,
        "final_x": float(info["x_position"]),
    }


def evaluate_run(run_dir, episodes, device):
    seed = parse_seed(run_dir)
    env = make_env(seed + EVAL_OFFSET)
    for ckpt in sorted(run_dir.glob("ckpt_ep*.zip")):
        ep_trained = int(re.search(r"ckpt_ep(\d+)", ckpt.name).group(1))
        model = SAC.load(ckpt.as_posix(), device=device)
        results = [rollout(model, env, seed + EVAL_OFFSET + k) for k in range(episodes)]
        data = {
            "seed": seed,
            "episodes_trained": ep_trained,
            "returns": np.array([r["return"] for r in results]),
            "successes": np.array([r["goal_reached"] for r in results]),
            "falls": np.array([r["is_fallen"] for r in results]),
            "lengths": np.array([r["length"] for r in results]),
            "final_x": np.array([r["final_x"] for r in results]),
        }
        np.savez(run_dir / f"eval_ep{ep_trained:04d}.npz", **data)
        print(f"[eval] seed={seed} ep{ep_trained:04d} "
              f"success={data['successes'].mean():.2f} "
              f"return={data['returns'].mean():7.1f} "
              f"falls={data['falls'].mean():.2f}")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="One or more run directories")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    run_dirs = sorted(Path(p) for p in args.run_dirs if Path(p).is_dir())
    if not run_dirs:
        raise SystemExit("no valid run directories given")
    for run_dir in run_dirs:
        evaluate_run(run_dir, args.episodes, args.device)


if __name__ == "__main__":
    main()