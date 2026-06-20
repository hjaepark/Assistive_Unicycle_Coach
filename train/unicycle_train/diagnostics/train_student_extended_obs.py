# train/unicycle_train/diagnostics/train_student_ref_execution.py

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import gymnasium as gym
import envs.unicycle
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from train.unicycle_train.diagnostics.train_student import REWARD, SAC_PARAMS


DATA_DIR = Path(__file__).resolve().parent / "data" / "student_extended_obs"

REF_MODEL_PATH = Path(__file__).resolve().parents[3] / (
    "train/unicycle_train/diagnostics/data/student/"
    "seed42_20260609_032834/student_agent.zip"
)


class EpisodeBudget(BaseCallback):
    def __init__(self, max_episodes):
        super().__init__()
        self.max_episodes = max_episodes
        self.n = 0

    def _on_step(self):
        self.n += sum(self.locals.get("dones", []))
        return self.n < self.max_episodes


class ReferenceExecutionWrapper(gym.Wrapper):
    def __init__(self, env, ref_model):
        super().__init__(env)
        self.ref_model = ref_model
        self.raw_obs = None
        self.prev_u_coach = np.zeros(self.action_space.shape, dtype=np.float32)

        raw_low = np.asarray(self.env.observation_space.low, dtype=np.float32)
        raw_high = np.asarray(self.env.observation_space.high, dtype=np.float32)

        action_low = np.asarray(self.action_space.low, dtype=np.float32)
        action_high = np.asarray(self.action_space.high, dtype=np.float32)

        coach_low = action_low - action_high
        coach_high = action_high - action_low

        aug_low = np.concatenate([raw_low, coach_low]).astype(np.float32)
        aug_high = np.concatenate([raw_high, coach_high]).astype(np.float32)

        self.observation_space = gym.spaces.Box(
            low=aug_low,
            high=aug_high,
            dtype=np.float32,
        )

        self.action_error_sum = 0.0
        self.u_coach_norm_sum = 0.0
        self.exec_ref_error_sum = 0.0
        self.action_count = 0

    def _make_aug_obs(self):
        return np.concatenate(
            [
                np.asarray(self.raw_obs, dtype=np.float32),
                np.asarray(self.prev_u_coach, dtype=np.float32),
            ]
        ).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw_obs = np.asarray(obs, dtype=np.float32)
        self.prev_u_coach = np.zeros(self.action_space.shape, dtype=np.float32)

        self.action_error_sum = 0.0
        self.u_coach_norm_sum = 0.0
        self.exec_ref_error_sum = 0.0
        self.action_count = 0

        return self._make_aug_obs(), info

    def step(self, student_action):
        ref_action, _ = self.ref_model.predict(self.raw_obs, deterministic=True)

        student_action = np.asarray(student_action, dtype=np.float32)
        ref_action = np.asarray(ref_action, dtype=np.float32)

        u_coach = ref_action - student_action

        exec_action = np.clip(
            student_action + u_coach,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)

        action_error = float(np.linalg.norm(student_action - ref_action))
        u_coach_norm = float(np.linalg.norm(u_coach))
        exec_ref_error = float(np.linalg.norm(exec_action - ref_action))

        self.action_error_sum += action_error
        self.u_coach_norm_sum += u_coach_norm
        self.exec_ref_error_sum += exec_ref_error
        self.action_count += 1

        obs, reward, terminated, truncated, info = self.env.step(exec_action)

        if terminated or truncated:
            n = max(self.action_count, 1)
            info["mean_action_error"] = self.action_error_sum / n
            info["mean_u_coach_norm"] = self.u_coach_norm_sum / n
            info["mean_exec_ref_error"] = self.exec_ref_error_sum / n

        self.prev_u_coach = u_coach.astype(np.float32)
        self.raw_obs = np.asarray(obs, dtype=np.float32)

        return self._make_aug_obs(), reward, terminated, truncated, info


def make_env(seed, run_dir, ref_model):
    env = gym.make("UnicyclePendulumPD-v0", reward_weights=REWARD, frame_skip=1)
    env = ReferenceExecutionWrapper(env, ref_model)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return Monitor(
        env,
        str(run_dir / "monitor"),
        info_keywords=(
            "goal_reached",
            "is_fallen",
            "mean_action_error",
            "mean_u_coach_norm",
            "mean_exec_ref_error",
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--data_dir", type=str, default=None)
    args = ap.parse_args()

    run_dir = (
        (Path(args.data_dir) if args.data_dir else DATA_DIR)
        / f"seed{args.seed:02d}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if not REF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Reference model not found: {REF_MODEL_PATH}")

    print(f"[ref] using reference model: {REF_MODEL_PATH}")

    ref_model = SAC.load(REF_MODEL_PATH.as_posix(), device=args.device)

    model = SAC(
        "MlpPolicy",
        make_env(args.seed, run_dir, ref_model),
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