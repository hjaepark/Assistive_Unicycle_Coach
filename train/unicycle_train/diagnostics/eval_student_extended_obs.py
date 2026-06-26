# train/unicycle_train/diagnostics/eval_student_extended_obs.py

import re
import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
import envs.unicycle
from stable_baselines3 import SAC

from train.unicycle_train.diagnostics.train_student import REWARD


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


def make_aug_obs(obs, action_dim):
    obs = np.asarray(obs, dtype=np.float32)
    zero_feedback = np.zeros(action_dim, dtype=np.float32)
    return np.concatenate([obs, zero_feedback]).astype(np.float32)


def rollout(model, env, seed):
    obs, info = env.reset(seed=seed)
    obs = make_aug_obs(obs, env.action_space.shape[0])

    done = False
    ep_return = 0.0
    length = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        obs = make_aug_obs(next_obs, env.action_space.shape[0])

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
    model = SAC.load((run_dir / "student_agent").as_posix(), device=device)

    env = make_env(seed + EVAL_OFFSET)
    results = [rollout(model, env, seed + EVAL_OFFSET + k) for k in range(episodes)]
    env.close()

    data = {
        "seed": seed,
        "returns": np.array([r["return"] for r in results]),
        "successes": np.array([r["goal_reached"] for r in results]),
        "falls": np.array([r["is_fallen"] for r in results]),
        "lengths": np.array([r["length"] for r in results]),
        "final_x": np.array([r["final_x"] for r in results]),
    }
    np.savez(run_dir / "eval.npz", **data)

    print(f"[eval] seed={seed:2d} "
          f"success={data['successes'].mean():.2f} "
          f"return={data['returns'].mean():7.1f} "
          f"falls={data['falls'].mean():.2f}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="One or more run directories")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    run_dirs = sorted(Path(p) for p in args.run_dirs if Path(p).is_dir())
    if not run_dirs:
        raise SystemExit("no valid run directories given")

    for run_dir in run_dirs:
        evaluate_run(run_dir, args.episodes, args.device)


if __name__ == "__main__":
    main()