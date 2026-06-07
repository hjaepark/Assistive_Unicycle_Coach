# train/unicycle_train/coach/v5/student_worker.py

"""
V5 Student Worker

Identical to V4 worker, plus loading the frozen autoencoder encoder
and dataset normalization stats. These are passed to CoachWrapper so
that get_pi_observation() can return [avg_reward, success_rate, z (32D)].

Each worker process owns:
  - One SAC student (specific seed)
  - One gymnasium env wrapped with V5 CoachWrapper
  - One π_A (BC pretrained CoachActor, frozen, loaded locally)
  - One frozen autoencoder encoder (loaded locally)
"""

import os
import numpy as np
import torch
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed as sb3_set_seed
from stable_baselines3.common.callbacks import BaseCallback

import envs.unicycle

from train.unicycle_train.coach.v5.coach_agent import CoachAgent
from train.unicycle_train.coach.v5.wrapper import CoachWrapper
from train.unicycle_train.coach.v5.autoencoder_model import TrajectoryAutoencoder


class StopAfterEpisodes(BaseCallback):
    """Stops student.learn() after N episodes."""
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


def _load_assistance_policy(bc_checkpoint, device="cpu"):
    """Load frozen π_A in this worker process."""
    coach = CoachAgent(
        obs_dim=22,
        action_dim=2,
        lr=0.0,
        device=device,
    )
    state_dict = torch.load(str(bc_checkpoint), map_location=device)
    coach.actor.load_state_dict(state_dict)
    coach.actor.eval()
    for p in coach.actor.parameters():
        p.requires_grad = False
    return coach


def _load_encoder(ae_checkpoint, device="cpu"):
    """Load frozen autoencoder encoder in this worker process."""
    ckpt = torch.load(str(ae_checkpoint), map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = TrajectoryAutoencoder(
        feature_dim=21,
        hidden_dim=cfg['hidden_dim'],
        latent_dim=cfg['latent_dim'],
        num_layers=cfg['num_layers'],
        max_seq_len=1500,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg['latent_dim']


def _load_norm_stats(dataset_processed_path):
    """Load z-score normalization stats from the training dataset."""
    data = np.load(dataset_processed_path)
    return (
        data['feature_mean'].astype(np.float32),
        data['feature_std'].astype(np.float32),
    )


def _make_env(log_dir, seed, reward_weights):
    base_env = gym.make(
        "UnicyclePendulumPD-v0",
        reward_weights=reward_weights,
    )
    monitored_env = Monitor(base_env, filename=str(log_dir / "monitor.csv"))
    monitored_env.reset(seed=seed)
    monitored_env.action_space.seed(seed)
    monitored_env.observation_space.seed(seed)
    return monitored_env


def _run_eval_phase(student, env, M, student_update=True):
    """Run M self-practice episodes. Mirrors V3 train.py logic."""
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


def _run_iteration(student, env, w_t, K, M, student_update_in_eval):
    """Execute one iteration: K coaching + M self-practice."""
    env.set_intervention(w_t)
    env.set_mode("coaching")

    coaching_stop = StopAfterEpisodes(max_episodes=K)

    obs, _ = env.reset()
    if student._last_obs is not None:
        student._last_obs = np.array([obs])

    student.learn(
        total_timesteps=10_000_000,
        callback=coaching_stop,
        reset_num_timesteps=False,
        progress_bar=False,
    )
    coaching_summary = env.get_phase_summary()

    eval_summary = _run_eval_phase(
        student, env, M,
        student_update=student_update_in_eval,
    )

    s_obs_next = env.get_pi_observation()
    coach_reward = eval_summary["avg_reward"]

    total_episodes = coaching_stop.episode_count + eval_summary["n_episodes"]
    total_steps = coaching_stop.step_count + eval_summary["total_steps"]

    return {
        "s_obs_next": s_obs_next.astype(np.float32),
        "coach_reward": float(coach_reward),
        "coaching_summary": coaching_summary,
        "eval_summary": eval_summary,
        "n_episodes": int(total_episodes),
        "n_steps": int(total_steps),
    }


def worker_main(pipe, worker_id, student_seed, bc_checkpoint, config, log_dir):
    """
    Entry point for one worker process. Runs forever until ("close",) received.
    """
    sb3_set_seed(student_seed)
    np.random.seed(student_seed)
    torch.manual_seed(student_seed)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load π_A (frozen)
    assistance_policy = _load_assistance_policy(
        bc_checkpoint,
        device=config["assistance_policy"]["device"],
    )

    # V5: Load frozen autoencoder encoder and normalization stats
    encoder, latent_dim = _load_encoder(
        config["autoencoder"]["checkpoint"],
        device=config["autoencoder"]["device"],
    )
    feature_mean, feature_std = _load_norm_stats(
        config["autoencoder"]["dataset_stats"],
    )

    # Create env + V5 wrapper (now takes encoder + norm stats)
    base_env = _make_env(log_dir, student_seed, config["reward_weights"])
    env = CoachWrapper(
        base_env,
        assistance_policy=assistance_policy,
        encoder=encoder,
        feature_mean=feature_mean,
        feature_std=feature_std,
        latent_dim=latent_dim,
        max_seq_len=1500,
        w_t=0,
    )

    # Create student SAC
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
        seed=student_seed,
    )

    K = config["training"]["K"]
    M = config["training"]["M"]
    student_update_in_eval = config["training"]["student_update_in_eval"]

    pipe.send({"status": "ready", "worker_id": worker_id, "seed": student_seed})

    try:
        while True:
            cmd = pipe.recv()
            if not isinstance(cmd, tuple) or len(cmd) != 2:
                pipe.send({"status": "error", "error": f"bad command: {cmd}"})
                continue

            cmd_type, cmd_arg = cmd

            if cmd_type == "step":
                w_t = int(cmd_arg)
                try:
                    result = _run_iteration(
                        student, env, w_t, K, M, student_update_in_eval,
                    )
                    result["status"] = "ok"
                    result["worker_id"] = worker_id
                    pipe.send(result)
                except Exception as e:
                    import traceback
                    pipe.send({
                        "status": "error",
                        "worker_id": worker_id,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })

            elif cmd_type == "save":
                save_path = Path(cmd_arg)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                student.save(save_path.as_posix())
                pipe.send({"status": "saved", "worker_id": worker_id, "path": str(save_path)})

            elif cmd_type == "close":
                if cmd_arg is not None:
                    save_path = Path(cmd_arg)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    student.save(save_path.as_posix())
                pipe.send({"status": "closed", "worker_id": worker_id})
                break

            else:
                pipe.send({
                    "status": "error",
                    "worker_id": worker_id,
                    "error": f"unknown command: {cmd_type}",
                })

    finally:
        try:
            env.close()
        except Exception:
            pass
        pipe.close()