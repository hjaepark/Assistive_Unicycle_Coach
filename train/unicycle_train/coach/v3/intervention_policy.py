# train/unicycle_train/coach/v3/intervention_policy.py

"""
V3 Intervention Policy (π_I)

Decides WHETHER to intervene (w_t ∈ {0, 1}) for the next coaching phase.
- Observation: [avg_episode_reward, success_rate] from previous coaching phase
- Action: discrete {0 = no intervention, 1 = intervene}
- Reward: student's independent eval performance (avg episode reward)
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

# InterventionActor outputs logits for binary action (0 or 1)
class InterventionActor(nn.Module):
    def __init__(self, obs_dim, hidden=[64, 64]):
        super().__init__()
        self.net = mlp(obs_dim, hidden, out_dim=2)

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

# Critic for value function V(s)
class InterventionCritic(nn.Module):
    def __init__(self, obs_dim, hidden=[64, 64]):
        super().__init__()
        self.net = mlp(obs_dim, hidden, out_dim=1)

    def forward(self, obs):
        return self.net(obs)

# InterventionPolicy encapsulates the PPO-based intervention policy π_I
class InterventionPolicy:
    def __init__(
        self,
        obs_dim=2, # [avg_reward, success_rate]
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,       
        entropy_coef=0.01,  
        vf_coef=0.5,        
        max_grad_norm=0.5, # Gradient clipping
        device="cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # Actor (categorical policy) and Critic (value function)
        self.actor = InterventionActor(obs_dim).to(self.device)
        self.critic = InterventionCritic(obs_dim).to(self.device)

        # Single optimizer for both actor and critic
        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        # Rollout buffer — stores transitions across rounds
        # Each "step" in π_I's perspective = one full round (K coaching + M eval)
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
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Get action
        if deterministic:
            logits = self.actor(obs)
            action = logits.argmax(dim=-1)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
        else:
            dist = self.actor.get_distribution(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Get value estimate
        value = self.critic(obs)

        return (
            int(action.squeeze().cpu().item()),
            float(log_prob.squeeze().cpu().item()),
            float(value.squeeze().cpu().item()),
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

        # Convert buffer to tensors
        obs = torch.as_tensor(np.array(buf["obs"]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buf["action"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(buf["log_prob"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(buf["reward"], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(buf["value"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(buf["done"], dtype=torch.float32, device=self.device)

        # Compute returns and advantages using GAE (simplified: no lambda)
        # Since each "step" is one round, we treat this as a simple trajectory
        returns = torch.zeros(n, device=self.device)
        running_return = 0.0
        for t in reversed(range(n)):
            if dones[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        advantages = returns - old_values
        # Normalize advantages (standard PPO practice)
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update over multiple epochs
        total_pi_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0

        for epoch in range(n_epochs):
            # New log probs and values
            new_log_probs = self.actor.log_prob(obs, actions)
            entropy = self.actor.entropy(obs)
            new_values = self.critic(obs).squeeze(-1)

            # Policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            pi_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            vf_loss = F.mse_loss(new_values, returns)

            # Entropy bonus (encourages exploration)
            entropy_loss = -entropy.mean()

            # Total loss
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

        # Clear buffer after update
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