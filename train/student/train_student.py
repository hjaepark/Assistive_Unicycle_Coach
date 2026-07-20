# train/student/train_student.py

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import envs.unicycle
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from train.student.recorder import Recorder

DATA_DIR = Path(__file__).resolve().parent / "data" / "student"

REWARD = {
    "posture": 0.0,
    "pendulum": 0.0,
    "speed_reward_weight": 0.0,
    "alive_bonus": 0.0,
    "upright_gate_k": 0.0,
    "progress": 60.0,
    "goal_bonus": 50.0,
    "fall_penalty": -20.0,
}

SAC_PARAMS = dict(
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=100,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
)

OBS_SCHEMA = [
    "pos_x", "pos_y", "pos_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "pendulum_angle", "wheel_angle",
    "vel_x", "vel_y", "vel_z",
    "angvel_x", "angvel_y", "angvel_z",
    "pendulum_angvel", "wheel_angvel",
    "pd_roll_torque", "pd_yaw_torque",
]

ACTION_SCHEMA = ["wheel_torque", "pitch_control_torque"]


class EpisodeBudget(BaseCallback):
    def __init__(self, max_episodes, checkpoint_every=0, run_dir=None):
        super().__init__()
        self.max_episodes = max_episodes
        self.checkpoint_episodes = (
            set(range(checkpoint_every, max_episodes + 1, checkpoint_every))
            if checkpoint_every > 0
            else set()
        )
        self.run_dir = run_dir
        self.n = 0

    def _on_step(self):
        done_count = sum(self.locals.get("dones", []))
        if done_count:
            self.n += done_count
            if self.n in self.checkpoint_episodes:
                ckpt = self.run_dir / f"ckpt_ep{self.n:04d}"
                self.model.save(ckpt.as_posix())
                self.model.save_replay_buffer(f"{ckpt.as_posix()}_buffer")
        return self.n < self.max_episodes


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


def save_meta(run_dir, args):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    meta = {
        "created": datetime.now().isoformat(),
        "seed": args.seed,
        "episodes": args.episodes,
        "checkpoint_every": args.checkpoint_every,
        "device": args.device,
        "reward": REWARD,
        "sac_params": SAC_PARAMS,
        "env_id": "UnicyclePendulumPD-v0",
        "frame_skip": 1,
        "obs_schema": OBS_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "git_commit": commit,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--checkpoint_every", type=int, default=50,
                    help="Save policy+buffer every N episodes (0 = off)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--data_dir", type=str, default=None)
    args = ap.parse_args()

    run_dir = (
        (Path(args.data_dir) if args.data_dir else DATA_DIR)
        / f"seed{args.seed:04d}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_meta(run_dir, args)

    model = SAC(
        "MlpPolicy",
        make_env(args.seed, run_dir),
        device=args.device,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        **SAC_PARAMS,
    )
    model.learn(
        total_timesteps=10_000_000,
        callback=EpisodeBudget(args.episodes, args.checkpoint_every, run_dir),
    )
    model.save((run_dir / "student_agent").as_posix())
    print(f"[done] seed={args.seed} episodes={args.episodes} saved to: {run_dir}")


if __name__ == "__main__":
    main()