# train/unicycle_train/coach/v3/train.py

"""
Coach V3 — Selective Intervention Training

Outer training loop:
  For each round:
    1. π_I observes student progress → decides w_t ∈ {0, 1}
    2. Coaching phase: student trains for K episodes with coach
    3. Eval phase: student runs M episodes alone (no update)
    4. π_I receives eval reward and stores transition
    5. Every update_every rounds, π_I does PPO update

Key differences from V0/V1/V2:
  - Coach reward = student's independent eval performance
  - Two-policy coach: π_I (intervention, PPO) + π_A (assistance, BC frozen)
  - Training alternates between coaching and eval phases

Usage:
    python -m train.unicycle_train.coach.v3.train
    python -m train.unicycle_train.coach.v3.train --config train/unicycle_train/coach/v3/configs/default.yaml
"""

import os
import argparse
import time
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed as sb3_set_seed
from stable_baselines3.common.callbacks import BaseCallback

import envs.unicycle

from train.unicycle_train.coach.v0.coach_agent import CoachAgent
from train.unicycle_train.coach.v3.intervention_policy import InterventionPolicy
from train.unicycle_train.coach.v3.wrapper import CoachWrapper

DEFAULT_CONFIG = {
    "experiment": {
        "name": "coach_v3",
    },

    "training": {
        "seed": 42,
        "K": 10,                    
        "M": 10,                    
        "max_rounds": 500,         
        "pi_update_every": 10,      
        "pi_ppo_epochs": 4,         
        "student_update_in_eval": True,  # False for V3, True for V3.1
    },

    "intervention_policy": {
        "obs_dim": 2,              
        "lr": 3e-4,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "entropy_coef": 0.01,
        "vf_coef": 0.5,
        "device": "cpu",
    },

    "assistance_policy": {
        "obs_dim": 22,              
        "action_dim": 2,
        "device": "cpu",
    },

    "student": {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "learning_starts": 1_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
    },

    "reward_weights": {
        "posture": 0.0,
        "pendulum": 0.0,
        "speed_reward_weight": 0.0,
        "alive_bonus": 0.0,
        "upright_gate_k": 0.0,
        "progress": 60.0,
        "goal_bonus": 50.0,
        "fall_penalty": -20.0,
    },
}

def load_config(config_path=None):
    config = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        for section, values in user_config.items():
            if section in config and isinstance(values, dict):
                config[section].update(values)
            else:
                config[section] = values
    return config

def find_latest_bc_checkpoint(v1_dir: Path) -> Path:
    logs_dir = v1_dir / "logs"
    bc_runs = sorted(logs_dir.glob("bc_*"))
    if not bc_runs:
        raise FileNotFoundError(f"No BC runs found in {logs_dir}")
    checkpoint = bc_runs[-1] / "checkpoints" / "best_bc_actor.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No best_bc_actor.pth in {bc_runs[-1]}")
    return checkpoint

def load_assistance_policy(bc_checkpoint: Path, config: dict) -> CoachAgent:
    coach = CoachAgent(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        lr=0.0,
        device=config["device"],
    )

    state_dict = torch.load(str(bc_checkpoint), map_location=config["device"])
    coach.actor.load_state_dict(state_dict)

    coach.actor.eval()
    for p in coach.actor.parameters():
        p.requires_grad = False

    return coach

def make_env(log_dir: Path, seed: int, reward_weights: dict):
    base_env = gym.make(
        "UnicyclePendulumPD-v0",
        reward_weights=reward_weights,
    )
    monitored_env = Monitor(base_env, filename=str(log_dir / "monitor.csv"))
    monitored_env.reset(seed=seed)
    monitored_env.action_space.seed(seed)
    monitored_env.observation_space.seed(seed)
    return monitored_env

class StopAfterEpisodes(BaseCallback):
    def __init__(self, max_episodes: int):
        super().__init__()
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.step_count = 0

    def _on_step(self):
        self.step_count += 1
        for done in self.locals["dones"]:
            if done:
                self.episode_count += 1
                if self.episode_count >= self.max_episodes:
                    return False
        return True

    def reset(self):
        self.episode_count = 0
        self.step_count = 0

def run_eval_phase(student, env, M: int, student_update: bool = False):
    env.set_mode("eval")

    if not student_update:
        episode_count = 0
        eval_steps = 0
        obs, info = env.reset()

        while episode_count < M:
            action, _ = student.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            eval_steps += 1

            if terminated or truncated:
                episode_count += 1
                if episode_count < M:
                    obs, info = env.reset()

        summary = env.get_phase_summary()
        summary["total_steps"] = eval_steps
        return summary

    else:
        eval_stop = StopAfterEpisodes(max_episodes=M)

        obs, _ = env.reset()
        if student._last_obs is not None:
            student._last_obs = np.array([obs])

        student.learn(
            total_timesteps=10_000_000,
            callback=eval_stop,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        summary = env.get_phase_summary()
        summary["total_steps"] = eval_stop.step_count
        return summary

def main():
    parser = argparse.ArgumentParser(description="Coach V3 Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bc_checkpoint", type=str, default=None,
                        help="Path to BC pretrained actor (auto-detects from V1 if omitted)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed

    SEED = config["training"]["seed"]
    sb3_set_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    script_dir = Path(__file__).resolve().parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = script_dir / "logs" / f"v3_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    with open(run_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    (run_dir / "seed.txt").write_text(f"{SEED}\n")

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    if args.bc_checkpoint is not None:
        bc_checkpoint = Path(args.bc_checkpoint)
    else:
        v1_dir = script_dir.parent / "v1"
        bc_checkpoint = find_latest_bc_checkpoint(v1_dir)

    if not bc_checkpoint.exists():
        print(f"[error] BC checkpoint not found: {bc_checkpoint}")
        exit(1)

    assistance_policy = load_assistance_policy(bc_checkpoint, config["assistance_policy"])
    print(f"[INIT] π_A loaded from: {bc_checkpoint}")

    intervention_policy = InterventionPolicy(
        obs_dim=config["intervention_policy"]["obs_dim"],
        lr=config["intervention_policy"]["lr"],
        gamma=config["intervention_policy"]["gamma"],
        clip_eps=config["intervention_policy"]["clip_eps"],
        entropy_coef=config["intervention_policy"]["entropy_coef"],
        vf_coef=config["intervention_policy"]["vf_coef"],
        device=config["intervention_policy"]["device"],
    )
    print(f"[INIT] π_I initialized (PPO, obs_dim=2, action={{0,1}})")

    env_log_dir = run_dir / "env"
    env_log_dir.mkdir(exist_ok=True)
    base_env = make_env(env_log_dir, SEED, config["reward_weights"])
    env = CoachWrapper(base_env, assistance_policy=assistance_policy, w_t=0)
    print(f"[INIT] Environment with V3 CoachWrapper created")

    student = SAC(
        "MlpPolicy",
        env,
        learning_rate=config["student"]["learning_rate"],
        buffer_size=config["student"]["buffer_size"],
        learning_starts=config["student"]["learning_starts"],
        batch_size=config["student"]["batch_size"],
        tau=config["student"]["tau"],
        gamma=config["student"]["gamma"],
        device="cpu",
        verbose=0,
        seed=SEED,
    )
    print(f"[INIT] Student (SAC) initialized")

    (run_dir / "bc_checkpoint.txt").write_text(str(bc_checkpoint))

    K = config["training"]["K"]
    M = config["training"]["M"]
    max_rounds = config["training"]["max_rounds"]
    update_every = config["training"]["pi_update_every"]
    student_update_in_eval = config["training"]["student_update_in_eval"]

    print()
    print("=" * 70)
    print("[INFO] Coach V3 — Selective Intervention Training")
    print("=" * 70)
    print(f"  Seed:             {SEED}")
    print(f"  K (coaching):     {K} episodes per round")
    print(f"  M (eval):         {M} episodes per round")
    print(f"  Max rounds:       {max_rounds}")
    print(f"  π_I update every: {update_every} rounds")
    print(f"  Student update in eval: {student_update_in_eval}")
    print(f"  BC checkpoint:    {bc_checkpoint}")
    print(f"  Run dir:          {run_dir}")
    print("=" * 70)
    print()

    training_log = []
    stop_callback = StopAfterEpisodes(max_episodes=K)

    pi_obs = np.array([0.0, 0.0], dtype=np.float32)

    total_episodes = 0
    total_steps = 0

    try:
        for iter_idx in range(1, max_rounds + 1):
            iter_start = time.time()

            w_t, log_prob, value = intervention_policy.predict(pi_obs)
            env.set_intervention(w_t)

            env.set_mode("coaching")
            stop_callback.reset()

            obs, _ = env.reset()
            if student._last_obs is not None:
                student._last_obs = np.array([obs])

            student.learn(
                total_timesteps=10_000_000,
                callback=stop_callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )

            coaching_summary = env.get_phase_summary()

            total_episodes += stop_callback.episode_count
            total_steps += stop_callback.step_count

            eval_summary = run_eval_phase(
                student, env, M,
                student_update=student_update_in_eval,
            )

            total_episodes += eval_summary["n_episodes"]
            total_steps += eval_summary["total_steps"]

            coach_reward = eval_summary["avg_reward"]

            pi_obs = env.get_pi_observation()

            intervention_policy.store_transition(
                obs=pi_obs,
                action=w_t,
                log_prob=log_prob,
                reward=coach_reward,
                value=value,
                done=False,
            )

            pi_logs = {}
            if iter_idx % update_every == 0:
                pi_logs = intervention_policy.train_step(
                    n_epochs=config["training"]["pi_ppo_epochs"]
                )

            iter_elapsed = time.time() - iter_start

            log_entry = {
                "iteration": iter_idx,
                "w_t": w_t,
                "coach_reward": coach_reward,
                "coaching_avg_reward": coaching_summary["avg_reward"],
                "coaching_success_rate": coaching_summary["success_rate"],
                "coaching_n_episodes": coaching_summary["n_episodes"],
                "eval_avg_reward": eval_summary["avg_reward"],
                "eval_success_rate": eval_summary["success_rate"],
                "eval_n_episodes": eval_summary["n_episodes"],
                "avg_coach_action_norm": coaching_summary["avg_coach_action_norm"],
                "total_episodes": total_episodes,
                "total_steps": total_steps,
                "wall_time": iter_elapsed,
                **pi_logs,
            }
            training_log.append(log_entry)

            # Tensorboard
            writer.add_scalar("coach/w_t", w_t, iter_idx)
            writer.add_scalar("coach/reward", coach_reward, iter_idx)
            writer.add_scalar("coaching/avg_reward", coaching_summary["avg_reward"], iter_idx)
            writer.add_scalar("coaching/success_rate", coaching_summary["success_rate"], iter_idx)
            writer.add_scalar("coaching/n_episodes", coaching_summary["n_episodes"], iter_idx)
            writer.add_scalar("eval/avg_reward", eval_summary["avg_reward"], iter_idx)
            writer.add_scalar("eval/success_rate", eval_summary["success_rate"], iter_idx)
            writer.add_scalar("eval/n_episodes", eval_summary["n_episodes"], iter_idx)
            writer.add_scalar("coaching/avg_coach_action_norm",
                              coaching_summary["avg_coach_action_norm"], iter_idx)
            writer.add_scalar("training/total_episodes", total_episodes, iter_idx)
            writer.add_scalar("training/total_steps", total_steps, iter_idx)
            for k, v in pi_logs.items():
                writer.add_scalar(f"pi_I/{k}", v, iter_idx)
            writer.flush()

            # Console output
            w_str = "INTERVENE" if w_t == 1 else "HANDS OFF"
            print(
                f"[Iter {iter_idx:4d}] w_t={w_str:10s} | "
                f"coaching: reward={coaching_summary['avg_reward']:+7.1f}, "
                f"success={coaching_summary['success_rate']*100:4.0f}% | "
                f"eval: reward={eval_summary['avg_reward']:+7.1f}, "
                f"success={eval_summary['success_rate']*100:4.0f}% | "
                f"episodes={total_episodes:6d}, steps={total_steps:8d} | "
                f"{iter_elapsed:.1f}s"
            )

            # Save periodically
            if iter_idx % 10 == 0:
                with open(run_dir / "training_log.json", 'w') as f:
                    json.dump(training_log, f, indent=2)

            if iter_idx % 50 == 0:
                intervention_policy.save(models_dir / f"pi_I_iter_{iter_idx:04d}.pth")
                student.save((models_dir / f"student_iter_{iter_idx:04d}").as_posix())

    except KeyboardInterrupt:
        print("\n\n[INFO] Training stopped by user (Ctrl+C)")

    finally:
        writer.close()

        intervention_policy.save(models_dir / "pi_I_final.pth")
        student.save((models_dir / "student_final").as_posix())

        with open(run_dir / "training_log.json", 'w') as f:
            json.dump(training_log, f, indent=2)

        print(f"\n[INFO] π_I saved: {models_dir / 'pi_I_final.pth'}")
        print(f"[INFO] Student saved: {models_dir / 'student_final'}")
        print(f"[INFO] Log saved: {run_dir / 'training_log.json'}")

        if training_log:
            rewards = [e["coach_reward"] for e in training_log]
            eval_success = [e["eval_success_rate"] for e in training_log]
            intervene_count = sum(1 for e in training_log if e["w_t"] == 1)

            print(f"\n{'='*70}")
            print(f"  Training Summary ({len(training_log)} iterations)")
            print(f"{'='*70}")
            print(f"  Total episodes:    {total_episodes}")
            print(f"  Total steps:       {total_steps}")
            print(f"  Student update in eval: {student_update_in_eval}")
            print(f"  Intervention rate: {intervene_count}/{len(training_log)} "
                  f"({100*intervene_count/len(training_log):.1f}%)")
            print(f"  Avg coach reward:  {np.mean(rewards):.2f}")
            print(f"  Avg eval success:  {np.mean(eval_success)*100:.1f}%")
            print(f"  Last 10 reward:    {np.mean(rewards[-10:]):.2f}")
            print(f"  Last 10 success:   {np.mean(eval_success[-10:])*100:.1f}%")
            print(f"{'='*70}")


if __name__ == "__main__":
    main()