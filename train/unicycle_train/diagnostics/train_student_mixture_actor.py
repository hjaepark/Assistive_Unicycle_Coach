# train/unicycle_train/diagnostics/train_student_mixture_actor.py

import argparse
from pathlib import Path
from datetime import datetime
from typing import NamedTuple

import numpy as np
import torch as th
import torch.nn.functional as F
import gymnasium as gym
import envs.unicycle
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update

from train.unicycle_train.diagnostics.train_student import REWARD, SAC_PARAMS


DATA_DIR = Path(__file__).resolve().parent / "data" / "student_mixture_actor"

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
    def __init__(self, env, ref_model, w=0.3):
        super().__init__(env)
        self.ref_model = ref_model
        self.w = w
        self.last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, student_action):
        ref_action, _ = self.ref_model.predict(self.last_obs, deterministic=True)
        student_action = np.asarray(student_action, dtype=np.float32)
        ref_action = np.asarray(ref_action, dtype=np.float32)

        exec_action = np.clip(
            student_action + self.w * (ref_action - student_action),
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)

        obs, reward, terminated, truncated, info = self.env.step(exec_action)
        info["exec_action"] = exec_action
        info["intervention_weight"] = self.w

        self.last_obs = obs
        return obs, reward, terminated, truncated, info


class MixtureReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    w: th.Tensor


class MixtureReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        pos = self.pos
        exec_actions = [info.get("exec_action", action[i]) for i, info in enumerate(infos)]
        action = np.asarray(exec_actions, dtype=action.dtype).reshape(action.shape)
        self.w[pos] = np.array(
            [info.get("intervention_weight", 0.0) for info in infos], dtype=np.float32
        )
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        base = super()._get_samples(batch_inds, env=env)
        w = self.to_torch(self.w[batch_inds, 0].reshape(-1, 1))
        return MixtureReplayBufferSamples(
            observations=base.observations,
            actions=base.actions,
            next_observations=base.next_observations,
            dones=base.dones,
            rewards=base.rewards,
            w=w,
        )


class MixtureActorSAC(SAC):
    def __init__(self, *args, ref_model_path=None, device="cuda", **kwargs):
        super().__init__(*args, device=device, **kwargs)
        ref = SAC.load(ref_model_path, device=device)
        self.ref_actor = ref.actor
        self.ref_actor.set_training_mode(False)
        for p in self.ref_actor.parameters():
            p.requires_grad_(False)

    def train(self, gradient_steps, batch_size=64):
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            with th.no_grad():
                a_coach = self.ref_actor._predict(replay_data.observations, deterministic=True)
            w = replay_data.w
            a_exec_for_actor = (1 - w) * actions_pi + w * a_coach

            q_values_pi = th.cat(self.critic(replay_data.observations, a_exec_for_actor), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


def make_env(seed, run_dir, ref_model, w):
    env = gym.make("UnicyclePendulumPD-v0", reward_weights=REWARD, frame_skip=1)
    env = ReferenceExecutionWrapper(env, ref_model, w)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return Monitor(
        env,
        str(run_dir / "monitor"),
        info_keywords=("goal_reached", "is_fallen", "intervention_weight"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--w", type=float, default=0.3)
    args = ap.parse_args()

    run_dir = (Path(args.data_dir) if args.data_dir else DATA_DIR) / f"seed{args.seed:02d}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not REF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Reference model not found: {REF_MODEL_PATH}")

    ref_model = SAC.load(REF_MODEL_PATH.as_posix(), device=args.device)

    model = MixtureActorSAC(
        "MlpPolicy",
        make_env(args.seed, run_dir, ref_model, args.w),
        replay_buffer_class=MixtureReplayBuffer,
        ref_model_path=REF_MODEL_PATH.as_posix(),
        device=args.device,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        **SAC_PARAMS,
    )

    model.learn(total_timesteps=10_000_000, callback=EpisodeBudget(args.episodes))
    model.save((run_dir / "student_agent").as_posix())

    print(f"[done] seed={args.seed} episodes={args.episodes} w={args.w} Saved to: {run_dir}")


if __name__ == "__main__":
    main()