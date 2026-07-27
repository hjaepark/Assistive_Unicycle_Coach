# train/coach/heuristic_2.py

import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym
import envs.unicycle
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from train.student.train_student import REWARD, EpisodeBudget
from train.student.recorder import Recorder
from train.student.baseline import find_start_ckpt, load_pool_seeds, START_EP

HEURISTIC_DIR = Path(__file__).resolve().parent / "data" / "heuristic_2"

COACH_EPISODES = 300
OBSERVE_EPISODES = 50
PROBE_EVERY = 10
PROBE_EPISODES = 5
COACH_ON_THRESHOLD = 0.20   
COACH_OFF_THRESHOLD = 0.60  
PROBE_SEED_OFFSET = 20000   


def run_probe(model, seed):
    env = gym.make("UnicyclePendulumPD-v0", reward_weights=REWARD, frame_skip=1)
    successes = 0
    for k in range(PROBE_EPISODES):
        obs, _ = env.reset(seed=seed + PROBE_SEED_OFFSET + k)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes += bool(info.get("goal_reached", False))
    env.close()
    return successes / PROBE_EPISODES


class HeuristicCoach2(gym.Wrapper):
    def __init__(self, env, expert):
        super().__init__(env)
        self.expert = expert
        self.probe_fn = None      # assigned after the model is created
        self.ep = 0
        self.successes = 0
        self.observe_rate = None
        self.coach_on = False
        self.intervened_episodes = 0
        self.probe_log = []       # [{"ep", "probe_rate", "coach_on"}, ...]
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs = obs
        return obs, info

    def step(self, action):
        if self.coach_on:
            coach_action, _ = self.expert.predict(self._obs, deterministic=True)
            action = np.clip(
                np.asarray(action) + np.asarray(coach_action),
                self.action_space.low,
                self.action_space.high,
            )
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs = obs
        info["coach_on"] = self.coach_on
        if self.coach_on:
            info["a_exec"] = np.asarray(action, dtype=np.float32)
        if terminated or truncated:
            self.ep += 1
            if self.coach_on:
                self.intervened_episodes += 1
            if self.ep <= OBSERVE_EPISODES:
                self.successes += bool(info.get("goal_reached", False))
            if self.ep == OBSERVE_EPISODES:
                self.observe_rate = self.successes / OBSERVE_EPISODES
            due = (self.ep >= OBSERVE_EPISODES
                   and (self.ep - OBSERVE_EPISODES) % PROBE_EVERY == 0
                   and self.ep < COACH_EPISODES)
            if due:
                rate = self.probe_fn()
                if rate < COACH_ON_THRESHOLD:
                    self.coach_on = True
                elif rate >= COACH_OFF_THRESHOLD:
                    self.coach_on = False
                # in between: hold the current state
                self.probe_log.append(
                    {"ep": self.ep, "probe_rate": rate, "coach_on": self.coach_on}
                )
        return obs, reward, terminated, truncated, info


class SharedControlSAC(SAC):
    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        a_exec = infos[0].get("a_exec")
        if a_exec is not None:
            buffer_action = self.policy.scale_action(a_exec.reshape(1, -1))
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)


def make_env(seed, run_dir, expert):
    env = gym.make("UnicyclePendulumPD-v0", reward_weights=REWARD, frame_skip=1)
    env = Recorder(env, run_dir / "episodes")
    env = HeuristicCoach2(env, expert)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    monitor = Monitor(
        env,
        str(run_dir / "monitor"),
        info_keywords=("goal_reached", "is_fallen", "coach_on"),
    )
    return monitor, env


def save_meta(run_dir, seed, device, expert_path):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    meta = {
        "created": datetime.now().isoformat(),
        "kind": "heuristic_2",
        "seed": seed,
        "resumed_from_ep": START_EP,
        "coach_episodes": COACH_EPISODES,
        "observe_episodes": OBSERVE_EPISODES,
        "probe_every": PROBE_EVERY,
        "probe_episodes": PROBE_EPISODES,
        "coach_on_threshold": COACH_ON_THRESHOLD,
        "coach_off_threshold": COACH_OFF_THRESHOLD,
        "probe_seed_offset": PROBE_SEED_OFFSET,
        "expert": str(expert_path),
        "device": device,
        "git_commit": commit,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def run_heuristic(seed, device, expert_path):
    ckpt, buf = find_start_ckpt(seed)

    run_dir = HEURISTIC_DIR / f"seed{seed:04d}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_meta(run_dir, seed, device, expert_path)

    expert = SAC.load(Path(expert_path).as_posix(), device=device)
    env, coach = make_env(seed, run_dir, expert)

    model = SharedControlSAC.load(ckpt.as_posix(), env=env, device=device)
    coach.probe_fn = lambda: run_probe(model, seed)
    model.tensorboard_log = str(run_dir / "tensorboard")
    model.load_replay_buffer(buf.with_suffix("").as_posix())

    budget = EpisodeBudget(
        max_episodes=COACH_EPISODES,
        checkpoint_every=COACH_EPISODES,
        run_dir=run_dir,
    )
    model.learn(
        total_timesteps=10_000_000,
        callback=budget,
        reset_num_timesteps=True,
    )
    model.save((run_dir / "student_agent").as_posix())

    ever = any(p["coach_on"] for p in coach.probe_log)
    first = next((p["ep"] for p in coach.probe_log if p["coach_on"]), None)
    result = {
        "ever_intervened": ever,
        "first_intervened_at_ep": first,
        "intervened_episode_count": coach.intervened_episodes,
        "observe_success_rate": coach.observe_rate,
        "n_probes": len(coach.probe_log),
        "probe_log": coach.probe_log,
    }
    (run_dir / "coach_result.json").write_text(json.dumps(result, indent=2))
    tag = (f"coach on from ep {first}, {coach.intervened_episodes} intervened eps"
           if ever else "never intervened")
    print(f"[heuristic_2] seed={seed} {tag} -> {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="Run a single seed (for cluster array jobs).")
    ap.add_argument("--expert", type=str, required=True,
                    help="Path to expert policy zip used as the coach action.")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    seeds = [args.seed] if args.seed is not None else load_pool_seeds()
    for s in seeds:
        run_heuristic(s, args.device, args.expert)


if __name__ == "__main__":
    main()