# train/unicycle_train/coach/v4/intervention_policy.py

"""
V4 Intervention Policy (π_I)

Identical to V3 — kept independent for V4 to avoid cross-version coupling.

Decides WHETHER to intervene (w_t ∈ {0, 1}) for the next coaching phase.
- Observation: [avg_episode_reward, success_rate] from previous phase
- Action: discrete {0 = no intervention, 1 = intervene}
- Reward: student's self-practice performance (avg episode reward)
- Algorithm: PPO (discrete categorical policy)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical


def mlp(in_dim, hidden=[64, 64], out_dim=0, act=nn.ReLU):
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    if out_dim > 0:
        layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class InterventionActor(nn.Module):
    """Outputs logits for n discrete actions.
    
    V4: n_actions=2 (binary {0, 1})
    V4.1: n_actions=5 (graduated assistance {0, 0.25, 0.5, 0.75, 1.0})
    V4.2 (future): n_actions=5 with w_values including negatives for resist
    """    
    def __init__(self, obs_dim, n_actions=5, hidden=[64, 64]):
        super().__init__()
        self.net = mlp(obs_dim, hidden, out_dim=n_actions)

    def forward(self, obs):
        return self.net(obs)

    def get_distribution(self, obs):
        logits = self.forward(obs)
        return Categorical(logits=logits)

    def log_prob(self, obs, action):
        dist = self.get_distribution(obs)
        return dist.log_prob(action)

    def entropy(self, obs):
        dist = self.get_distribution(obs)
        return dist.entropy()


class InterventionCritic(nn.Module):
    """Value function V(s)."""
    def __init__(self, obs_dim, hidden=[64, 64]):
        super().__init__()
        self.net = mlp(obs_dim, hidden, out_dim=1)

    def forward(self, obs):
        return self.net(obs)


class InterventionPolicy:
    def __init__(
        self,
        obs_dim=2,
        n_actions=5,                    # V4.1 NEW: number of discrete intervention levels
        w_values=None,                  # V4.1 NEW: list mapping action index -> actual w_t scalar
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,
        entropy_coef=0.02,              # V4.1: raised from 0.01 to encourage exploring all 5 actions
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # V4.1 NEW: store action-to-w mapping
        # Default is graduated assistance evenly spaced from 0 to 1
        # For V4.2 resist, pass w_values=[-1.0, -0.5, 0.0, 0.5, 1.0] explicitly
        self.n_actions = n_actions
        if w_values is None:
            w_values = np.linspace(0.0, 1.0, n_actions).tolist()
        assert len(w_values) == n_actions, (
            f"w_values length ({len(w_values)}) must match n_actions ({n_actions})"
        )
        self.w_values = list(w_values)

        # Pass n_actions to actor; critic is unchanged (still outputs scalar V(s))
        self.actor = InterventionActor(obs_dim, n_actions=n_actions).to(self.device)
        self.critic = InterventionCritic(obs_dim).to(self.device)

        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        self.buffer = {
            "obs": [],
            "action": [],
            "log_prob": [],
            "reward": [],
            "value": [],
            "done": [],
        }

    @torch.no_grad()
    def predict(self, obs_np, deterministic=False):
        """Returns (action_idx, w_t, log_prob, value) for a SINGLE observation.
        
        V4.1 CHANGE: now returns 4 values instead of 3.
        - action_idx: integer index in [0, n_actions) used for PPO buffer/loss
        - w_t: float scalar in self.w_values used by the wrapper as actual coaching weight
        
        The two are kept separate because PPO's categorical log_prob is defined
        on the index space, while the environment needs the actual scalar coefficient.
        """
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if deterministic:
            logits = self.actor(obs)
            action = logits.argmax(dim=-1)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
        else:
            dist = self.actor.get_distribution(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        value = self.critic(obs)

        action_idx = int(action.squeeze().cpu().item())
        # V4.1 NEW: map discrete index to actual w_t scalar
        w_t = self.w_values[action_idx]

        return (
            action_idx,                               # for PPO buffer (integer)
            w_t,                                      # for wrapper (float)
            float(log_prob.squeeze().cpu().item()),
            float(value.squeeze().cpu().item()),
        )

    @torch.no_grad()
    def predict_batch(self, obs_batch_np, deterministic=False):
        """Batched prediction for N students.
        
        V4.1 CHANGE: returns (action_idxs, w_ts, log_probs, values) — 4 lists instead of 3.
        """
        obs = torch.as_tensor(obs_batch_np, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if deterministic:
            logits = self.actor(obs)
            actions = logits.argmax(dim=-1)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
        else:
            dist = self.actor.get_distribution(obs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        values = self.critic(obs).squeeze(-1)

        action_idxs = actions.cpu().numpy().astype(int).tolist()
        # V4.1 NEW: map each index to its corresponding w_t scalar
        w_ts = [self.w_values[idx] for idx in action_idxs]

        return (
            action_idxs,                              # for PPO buffer (list of ints)
            w_ts,                                     # for wrappers (list of floats)
            log_probs.cpu().numpy().astype(float).tolist(),
            values.cpu().numpy().astype(float).tolist(),
        )

    def store_transition(self, obs, action, log_prob, reward, value, done=False):
        self.buffer["obs"].append(np.array(obs, dtype=np.float32))
        self.buffer["action"].append(action)
        self.buffer["log_prob"].append(log_prob)
        self.buffer["reward"].append(reward)
        self.buffer["value"].append(value)
        self.buffer["done"].append(done)

    def train_step(self, n_epochs=4):
        buf = self.buffer
        n = len(buf["obs"])

        if n == 0:
            return {"train/pi_loss": 0.0}

        obs = torch.as_tensor(np.array(buf["obs"]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buf["action"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(buf["log_prob"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(buf["reward"], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(buf["value"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(buf["done"], dtype=torch.float32, device=self.device)

        # Compute returns (simple cumulative, no GAE)
        returns = torch.zeros(n, device=self.device)
        running_return = 0.0
        for t in reversed(range(n)):
            if dones[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        advantages = returns - old_values
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_pi_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0

        for epoch in range(n_epochs):
            new_log_probs = self.actor.log_prob(obs, actions)
            entropy = self.actor.entropy(obs)
            new_values = self.critic(obs).squeeze(-1)

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            pi_loss = -torch.min(surr1, surr2).mean()

            vf_loss = F.mse_loss(new_values, returns)
            entropy_loss = -entropy.mean()

            loss = pi_loss + self.vf_coef * vf_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            total_pi_loss += pi_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy.mean().item()

        for key in self.buffer:
            self.buffer[key] = []

        return {
            "train/pi_loss": total_pi_loss / n_epochs,
            "train/vf_loss": total_vf_loss / n_epochs,
            "train/entropy": total_entropy / n_epochs,
            "train/mean_advantage": float(advantages.mean().item()),
            "train/mean_return": float(returns.mean().item()),
            "train/buffer_size": n,
        }

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])