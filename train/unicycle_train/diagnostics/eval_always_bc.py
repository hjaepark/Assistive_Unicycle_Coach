# train/unicycle_train/diagnostics/eval_always_bc.py
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import gymnasium as gym
import envs.unicycle
from stable_baselines3 import SAC
from train.unicycle_train.diagnostics.train_student import REWARD, SAC_PARAMS
from train.unicycle_train.diagnostics.rollout import rollout
from train.unicycle_train.diagnostics.bc_io import load_bc_actor, predict_delta

DATA_DIR = Path(__file__).resolve().parent / "data" / "always_bc"
EVAL_OFFSET = 10000


def make_env(seed):
    env = gym.make("UnicyclePendulumPD-v0", reward_weights=REWARD, frame_skip=1)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_policy(student, actor, device, reward_zero):
    def policy_fn(obs, prev_reward):
        a_student, _ = student.predict(obs, deterministic=True)
        r = 0.0 if reward_zero else prev_reward
        delta = predict_delta(actor, obs, a_student, r, device)
        a_exec = np.clip(a_student + delta, -1.0, 1.0)
        return a_exec, {"sat": float(np.abs(delta).max() > 0.95)}
    return policy_fn


def evaluate_seed(seed, episodes, actor, device, ckpt, reward_zero):
    env = make_env(seed + EVAL_OFFSET)
    student = SAC("MlpPolicy", env, device=device, seed=seed, **SAC_PARAMS)
    policy_fn = make_policy(student, actor, device, reward_zero)
    results = [rollout(policy_fn, env, seed + EVAL_OFFSET + k) for k in range(episodes)]
    env.close()

    data = {
        "seed": seed,
        "checkpoint": str(ckpt),
        "reward_zero": reward_zero,
        "returns": np.array([r["return"] for r in results]),
        "successes": np.array([r["goal_reached"] for r in results]),
        "falls": np.array([r["is_fallen"] for r in results]),
        "lengths": np.array([r["length"] for r in results]),
        "final_x": np.array([r["final_x"] for r in results]),
        "sat": np.array([r["sat"] for r in results]),
    }
    run_dir = DATA_DIR / f"seed{seed:02d}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "eval.npz", **data)
    print(f"[always_bc] seed={seed:2d} "
          f"success={data['successes'].mean():.2f} "
          f"final_x={data['final_x'].mean():5.2f} "
          f"falls={data['falls'].mean():.2f} "
          f"sat={data['sat'].mean():.2f}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--reward_zero", action="store_true")
    args = ap.parse_args()

    actor, ckpt = load_bc_actor(args.device, args.checkpoint)
    print(f"[bc] {ckpt}")
    for seed in args.seeds:
        evaluate_seed(seed, args.episodes, actor, args.device, ckpt, args.reward_zero)


if __name__ == "__main__":
    main()