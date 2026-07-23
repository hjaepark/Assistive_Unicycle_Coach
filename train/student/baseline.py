# train/student/baseline.py

import csv
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import envs.unicycle
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from train.student.train_student import REWARD, EpisodeBudget
from train.student.recorder import Recorder

STUDENT_DIR = Path(__file__).resolve().parent / "data" / "student"
BASELINE_DIR = Path(__file__).resolve().parent / "data" / "baseline"
POOL_CSV = Path(__file__).resolve().parent / "data" / "pool_ep400.csv"

START_EP = 400
BASELINE_EPISODES = 300 #500 Too long - 60% student converges


def find_start_ckpt(seed):
    matches = sorted(STUDENT_DIR.glob(f"seed{seed:04d}_*"))
    if not matches:
        raise FileNotFoundError(f"no run dir for seed {seed} under {STUDENT_DIR}")
    run_dir = matches[0]
    ckpt = run_dir / f"ckpt_ep{START_EP:04d}.zip"
    buf = run_dir / f"ckpt_ep{START_EP:04d}_buffer.pkl"
    if not ckpt.exists() or not buf.exists():
        raise FileNotFoundError(f"missing ep{START_EP} ckpt/buffer in {run_dir}")
    return ckpt, buf


def make_env(seed, run_dir):
    env = gym.make("UnicyclePendulumPD-v0", reward_weights=REWARD, frame_skip=1)
    env = Recorder(env, run_dir / "episodes")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return Monitor(
        env,
        str(run_dir / "monitor"),
        info_keywords=("goal_reached", "is_fallen"),
    )


def save_meta(run_dir, seed, device):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    meta = {
        "created": datetime.now().isoformat(),
        "kind": "baseline",
        "seed": seed,
        "resumed_from_ep": START_EP,
        "baseline_episodes": BASELINE_EPISODES,
        "device": device,
        "git_commit": commit,
    }
    (run_dir / "meta.json").write_text(__import__("json").dumps(meta, indent=2))


def run_baseline(seed, device):
    ckpt, buf = find_start_ckpt(seed)

    run_dir = BASELINE_DIR / f"seed{seed:04d}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_meta(run_dir, seed, device)

    env = make_env(seed, run_dir)

    model = SAC.load(ckpt.as_posix(), env=env, device=device)
    model.tensorboard_log = str(run_dir / "tensorboard")
    model.load_replay_buffer(buf.with_suffix("").as_posix())

    budget = EpisodeBudget(
        max_episodes=BASELINE_EPISODES,
        checkpoint_every=BASELINE_EPISODES,
        run_dir=run_dir,
    )
    model.learn(
        total_timesteps=10_000_000,
        callback=budget,
        reset_num_timesteps=True,
    )
    model.save((run_dir / "student_agent").as_posix())
    print(f"[baseline] seed={seed} +{BASELINE_EPISODES}ep from ep{START_EP} -> {run_dir}")


def load_pool_seeds():
    with open(POOL_CSV) as f:
        return [int(row["seed"]) for row in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="Run a single seed (for cluster array jobs).")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    seeds = [args.seed] if args.seed is not None else load_pool_seeds()
    for s in seeds:
        run_baseline(s, args.device)


if __name__ == "__main__":
    main()