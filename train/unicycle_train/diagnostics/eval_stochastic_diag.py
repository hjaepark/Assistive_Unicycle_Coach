# train/unicycle_train/diagnostics/eval_stochastic_diag.py
#
# One-off DIAGNOSTIC (not a protocol change): re-evaluates runs with the
# STOCHASTIC policy on the same eval seeds, and compares against the saved
# deterministic eval.npz.
#
# Hypothesis under test: on collapsed-but-healthy-tail seeds (e.g. 33, 85),
# the Gaussian actor mean straddles two Q modes (expert demos vs student's
# own style), so the deterministic mean action fails while sampled actions
# succeed. Prediction: stochastic success >> deterministic success.
#
# Usage:
#   python -m train.unicycle_train.diagnostics.eval_stochastic_diag \
#       data/student_demo_inject/seed33_* data/student_demo_inject/seed85_*

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


def rollout(model, env, seed, deterministic):
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return bool(info["goal_reached"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    for run_dir in sorted(p for p in args.run_dirs if p.is_dir()):
        seed = parse_seed(run_dir)
        model = SAC.load((run_dir / "student_agent").as_posix(), device=args.device)
        env = make_env(seed + EVAL_OFFSET)

        stoch = np.mean([
            rollout(model, env, seed + EVAL_OFFSET + k, deterministic=False)
            for k in range(args.episodes)
        ])
        env.close()

        det = float("nan")
        npz = run_dir / "eval.npz"
        if npz.exists():
            det = float(np.load(npz)["successes"].mean())

        gap = stoch - det if not np.isnan(det) else float("nan")
        print(f"[diag] seed={seed:2d}  deterministic={det:.2f}  "
              f"stochastic={stoch:.2f}  gap={gap:+.2f}")


if __name__ == "__main__":
    main()