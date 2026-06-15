# train/unicycle_train/diagnostics/train_student.py

import argparse
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import envs.unicycle
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


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


class EpisodeBudget(BaseCallback):
    def __init__(self, max_episodes):
        super().__init__()
        self.max_episodes = max_episodes
        self.n = 0

    def _on_step(self):
        self.n += sum(self.locals.get("dones", []))
        return self.n < self.max_episodes


def make_env(seed, run_dir):
    env = gym.make("UnicyclePendulumPD-v0", reward_weights=REWARD, frame_skip=1)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return Monitor(env, str(run_dir / "monitor"), info_keywords=("goal_reached", "is_fallen"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--data_dir", type=str, default=None)
    args = ap.parse_args()

    run_dir = (Path(args.data_dir) if args.data_dir else DATA_DIR) / f"seed{args.seed:02d}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = SAC(
        "MlpPolicy",
        make_env(args.seed, run_dir),
        device=args.device,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        **SAC_PARAMS,
    )
    model.learn(total_timesteps=10_000_000, callback=EpisodeBudget(args.episodes))
    model.save((run_dir / "student_agent").as_posix())

    print(f"[done] seed={args.seed} episodes={args.episodes} Saved to: {run_dir}")


if __name__ == "__main__":
    main()