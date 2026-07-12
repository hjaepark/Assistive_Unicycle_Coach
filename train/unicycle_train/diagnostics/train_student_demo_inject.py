# train/unicycle_train/diagnostics/train_student_demo_inject.py
#
# Episode-level demo injection (experiments A1 / A2 / A3).
#
# Design principles:
#   1. Student is PLAIN SB3 SAC. No custom train(), no gradient scaling.
#   2. Control authority is binary per EPISODE: an episode is either fully
#      coach-driven (demo) or fully student-driven. No action mixing exists.
#   3. The replay buffer always stores the action that actually ran in the
#      environment, so causal attribution / Bellman consistency is honest.
#   4. Demo episodes count against the total episode budget (fair accounting).
#
# Conditions:
#   A1: --prefix_demos 10                 (first 10 episodes are demos)
#   A2: --demo_every 5                    (every 5th episode is a demo)
#   A3: --demo_every 10                   (every 10th episode is a demo)

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import gymnasium as gym
import envs.unicycle
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from train.unicycle_train.diagnostics.train_student import REWARD, SAC_PARAMS


DATA_DIR = Path(__file__).resolve().parent / "data" / "student_demo_inject"

REF_MODEL_PATH = Path(__file__).resolve().parents[3] / (
    "train/unicycle_train/diagnostics/data/student/"
    "seed42_20260609_032834/student_agent.zip"
)


class EpisodeBudget(BaseCallback):
    """Stop training after max_episodes total episodes (demos included)."""

    def __init__(self, max_episodes):
        super().__init__()
        self.max_episodes = max_episodes
        self.n = 0

    def _on_step(self):
        self.n += sum(self.locals.get("dones", []))
        return self.n < self.max_episodes


class CoachDemoWrapper(gym.Wrapper):
    """Per-episode control authority.

    On demo episodes the coach (reference expert) drives every step and the
    student's proposed actions are ignored. On all other episodes the student
    drives. The executed action is exposed via info["exec_action"] so the
    replay buffer can store what actually happened.
    """

    def __init__(self, env, ref_model, demo_every=None, prefix_demos=0):
        super().__init__(env)
        self.ref_model = ref_model
        self.demo_every = demo_every
        self.prefix_demos = prefix_demos
        self.ep_idx = -1
        self.is_demo = False
        self.last_obs = None

    def _episode_is_demo(self, ep_idx):
        if ep_idx < self.prefix_demos:
            return True
        if self.demo_every is not None and ep_idx % self.demo_every == 0:
            return True
        return False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.ep_idx += 1
        self.is_demo = self._episode_is_demo(self.ep_idx)
        self.last_obs = obs
        return obs, info

    def step(self, student_action):
        if self.is_demo:
            exec_action, _ = self.ref_model.predict(self.last_obs, deterministic=True)
        else:
            exec_action = student_action
        exec_action = np.asarray(exec_action, dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(exec_action)

        # Always record what actually ran -> buffer stays causally honest.
        info["exec_action"] = exec_action
        info["coach_episode"] = self.is_demo

        self.last_obs = obs
        return obs, reward, terminated, truncated, info


class ExecActionReplayBuffer(ReplayBuffer):
    """ReplayBuffer that stores the executed action instead of the policy output.

    On student episodes exec_action == student_action, so this is a no-op.
    On demo episodes it swaps in the coach action, keeping (s, a, r, s')
    consistent with the true environment transition.
    """

    def add(self, obs, next_obs, action, reward, done, infos):
        exec_actions = [
            info.get("exec_action", action[i]) for i, info in enumerate(infos)
        ]
        action = np.asarray(exec_actions, dtype=action.dtype).reshape(action.shape)
        super().add(obs, next_obs, action, reward, done, infos)


def make_env(seed, run_dir, ref_model, demo_every, prefix_demos):
    env = gym.make("UnicyclePendulumPD-v0", reward_weights=REWARD, frame_skip=1)
    # Seed BEFORE wrapping: otherwise this reset passes through the wrapper
    # and consumes ep_idx 0, shifting the whole demo schedule by one.
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env = CoachDemoWrapper(env, ref_model, demo_every=demo_every, prefix_demos=prefix_demos)
    # coach_episode in the final-step info lets Monitor tag each episode,
    # so demo vs student episodes can be separated in analysis.
    return Monitor(
        env,
        str(run_dir / "monitor"),
        info_keywords=("goal_reached", "is_fallen", "coach_episode"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--demo_every", type=int, default=None,
                    help="every N-th episode is a full coach demo (A2/A3)")
    ap.add_argument("--prefix_demos", type=int, default=0,
                    help="first N episodes are full coach demos (A1)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--data_dir", type=str, default=None)
    args = ap.parse_args()

    if args.demo_every is None and args.prefix_demos == 0:
        raise ValueError("Set --demo_every (A2/A3) or --prefix_demos (A1); "
                         "otherwise this is just the solo baseline.")

    tag = (f"demoevery{args.demo_every}" if args.demo_every is not None
           else f"prefix{args.prefix_demos}")
    run_dir = (Path(args.data_dir) if args.data_dir else DATA_DIR) / (
        f"seed{args.seed:02d}_{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if not REF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Reference model not found: {REF_MODEL_PATH}")

    ref_model = SAC.load(REF_MODEL_PATH.as_posix(), device=args.device)

    model = SAC(
        "MlpPolicy",
        make_env(args.seed, run_dir, ref_model, args.demo_every, args.prefix_demos),
        replay_buffer_class=ExecActionReplayBuffer,
        device=args.device,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        **SAC_PARAMS,
    )

    model.learn(total_timesteps=10_000_000, callback=EpisodeBudget(args.episodes))
    model.save((run_dir / "student_agent").as_posix())

    print(f"[done] seed={args.seed} episodes={args.episodes} "
          f"demo_every={args.demo_every} prefix_demos={args.prefix_demos} "
          f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()