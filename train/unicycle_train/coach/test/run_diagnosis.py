# train/unicycle_train/coach/test/run_diagnosis.py
"""
Buffer-mislabeling diagnosis (minimal: fixed-rule coach, no pi_I learning, no z).

Question:
  A strong student that reaches high solo success collapses when trained under
  coach intervention. Is the cause the student replay buffer storing
  (student_action, reward_from_exec_action) -- mislabeled transitions?

Design (two variables + seed):
  Train ONE SAC student for --train-episodes under a FIXED-RULE coach
  (intervene each step with prob --intervene-rate; frozen pi_A gives delta).
  No phases. Then remove the coach and evaluate the student alone.

  --buffer-mode mislabel : SB3 default. action=student_action, outcome=exec. (V5)
  --buffer-mode honest   : relabel action -> exec_action (TS2C-style absorption)
  --buffer-mode exclude  : skip intervened steps (student learns own experience)

  --intervene-rate 1.0 / 0.5 / 0.25 : fraction of steps the coach intervenes.
  --intervene-rate 0.0              : solo baseline (no coach).

Compare post_removal_success across the grid.
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed as sb3_set_seed
from stable_baselines3.common.callbacks import BaseCallback

import envs.unicycle  # registers UnicyclePendulumPD-v0

from train.unicycle_train.coach.v5.coach_agent import CoachAgent
from train.unicycle_train.coach.test.mini_wrapper import MiniCoachWrapper


REWARD_WEIGHTS = {
    "posture": 0.0, "pendulum": 0.0, "speed_reward_weight": 0.0,
    "alive_bonus": 0.0, "upright_gate_k": 0.0,
    "progress": 60.0, "goal_bonus": 50.0, "fall_penalty": -20.0,
}
STUDENT_CONFIG = {
    "learning_rate": 3e-4, "buffer_size": 1_000_000, "learning_starts": 1_000,
    "batch_size": 256, "tau": 0.005, "gamma": 0.99,
}


class DiagnosisReplayBuffer(ReplayBuffer):
    """Relabel or skip intervened transitions, based on env-provided info."""
    def __init__(self, *args, buffer_mode="mislabel", **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_mode = buffer_mode

    def add(self, obs, next_obs, action, reward, done, infos):
        info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) else {}
        w_t = int(info0.get("w_t", 0))

        if self.buffer_mode == "exclude" and w_t == 1:
            return  # drop intervened step

        if self.buffer_mode == "honest" and w_t == 1 and "exec_action" in info0:
            exec_a = np.asarray(info0["exec_action"], dtype=np.asarray(action).dtype)
            action = exec_a.reshape(np.asarray(action).shape)

        super().add(obs, next_obs, action, reward, done, infos)


class StopAfterEpisodes(BaseCallback):
    def __init__(self, max_episodes):
        super().__init__()
        self.max_episodes = max_episodes
        self.count = 0

    def _on_step(self):
        for done in self.locals["dones"]:
            if done:
                self.count += 1
        return self.count < self.max_episodes


class EpisodeSuccessLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.successes = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.successes.append(1.0 if info.get("goal_reached", False) else 0.0)
        return True


def load_assistance_policy(bc_checkpoint, device="cpu"):
    coach = CoachAgent(obs_dim=22, action_dim=2, lr=0.0, device=device)
    sd = torch.load(str(bc_checkpoint), map_location=device)
    coach.actor.load_state_dict(sd)
    coach.actor.eval()
    for p in coach.actor.parameters():
        p.requires_grad = False
    return coach


def find_latest_bc_checkpoint(coach_dir: Path) -> Path:
    logs_dir = coach_dir / "v1" / "logs"
    bc_runs = sorted(logs_dir.glob("bc_*"))
    if not bc_runs:
        raise FileNotFoundError(f"No BC runs in {logs_dir}")
    ckpt = bc_runs[-1] / "checkpoints" / "best_bc_actor.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"No best_bc_actor.pth in {bc_runs[-1]}")
    return ckpt


def make_base_env(log_dir, seed):
    env = gym.make("UnicyclePendulumPD-v0", reward_weights=REWARD_WEIGHTS)
    env = Monitor(env, filename=str(log_dir / "monitor.csv"))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True,
                    help="Student identity (environment reset seed)")
    ap.add_argument("--rep-seed", type=int, default=None,
                    help="Repetition seed for SAC/torch init; defaults to --seed")
    ap.add_argument("--buffer-mode", choices=["mislabel", "honest", "exclude"],
                    default="mislabel")
    ap.add_argument("--intervene-rate", type=float, required=True,
                    help="0.0=solo baseline, 1.0=always intervene")
    ap.add_argument("--train-episodes", type=int, default=500,
                    help="Matches eval Phase A")
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--bc-checkpoint", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default="results")
    args = ap.parse_args()

    rep_seed = args.rep_seed if args.rep_seed is not None else args.seed
    sb3_set_seed(rep_seed); np.random.seed(rep_seed); torch.manual_seed(rep_seed)

    script_dir = Path(__file__).resolve().parent
    coach_dir = script_dir.parent
    is_solo = (args.intervene_rate <= 0.0)
    tag = "solo" if is_solo else f"r{args.intervene_rate}_{args.buffer_mode}"
    out_dir = Path(args.out_dir) / f"seed{args.seed}_{tag}_rep{rep_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pi_A = None
    if not is_solo:
        bc_ckpt = (Path(args.bc_checkpoint) if args.bc_checkpoint
                   else find_latest_bc_checkpoint(coach_dir))
        pi_A = load_assistance_policy(bc_ckpt, device=args.device)
        print(f"[cfg] pi_A = {bc_ckpt}")
    print(f"[cfg] seed={args.seed} rate={args.intervene_rate} "
          f"mode={'-' if is_solo else args.buffer_mode} "
          f"train_eps={args.train_episodes}")

    # ---- Phase A: train under fixed-rule coach -------------------------
    train_env = MiniCoachWrapper(
        make_base_env(out_dir, args.seed), assistance_policy=pi_A,
        intervene_rate=(0.0 if is_solo else args.intervene_rate),
        rng_seed=args.seed,
    )
    student = SAC(
        "MlpPolicy", train_env,
        replay_buffer_class=DiagnosisReplayBuffer,
        replay_buffer_kwargs={"buffer_mode": args.buffer_mode},
        device=args.device, verbose=0, seed=rep_seed,
        **STUDENT_CONFIG,
    )

    t0 = time.time()
    student.learn(total_timesteps=10_000_000,
                  callback=StopAfterEpisodes(args.train_episodes),
                  progress_bar=False)
    train_time = time.time() - t0
    train_env.close()

    # ---- Phase B: remove coach, evaluate student alone -----------------
    eval_env = MiniCoachWrapper(
        make_base_env(out_dir, args.seed + 10_000), assistance_policy=pi_A,
        intervene_rate=0.0,  # coach OFF
        rng_seed=args.seed + 10_000,
    )
    student.set_env(eval_env)
    succ = EpisodeSuccessLogger()
    student.learn(total_timesteps=10_000_000,
                  callback=[succ, StopAfterEpisodes(args.eval_episodes)],
                  reset_num_timesteps=False, progress_bar=False)
    eval_env.close()

    post = float(np.mean(succ.successes)) if succ.successes else 0.0
    result = {
        "seed": args.seed, "rep_seed": rep_seed, "intervene_rate": args.intervene_rate,
        "buffer_mode": ("solo" if is_solo else args.buffer_mode),
        "train_episodes": args.train_episodes, "train_time_s": train_time,
        "eval_episodes": len(succ.successes), "post_removal_success": post,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] seed={args.seed} rate={args.intervene_rate} "
          f"mode={result['buffer_mode']} post_removal={post*100:.1f}%")


if __name__ == "__main__":
    main()