# train/unicycle_train/coach/v5/coach_agent.py

"""
V4 Coach Agent — used for π_A (assistance policy) only.

Identical to V0 CoachAgent. Kept independent for V4.
In V4 training, π_A is loaded from BC checkpoint and frozen.
"""

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

LOG_STD_MIN, LOG_STD_MAX = -5, 2


def mlp(in_dim, hidden=[256, 256], out_dim=0, act=nn.ReLU):
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    if out_dim > 0:
        layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class CoachActor(nn.Module):
    """Tanh-normal policy that outputs 2D assist action."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.body = mlp(obs_dim, [256, 256])
        self.mu_head = nn.Linear(256, action_dim)
        self.logstd_head = nn.Linear(256, action_dim)

    def forward(self, obs):
        z = self.body(obs)
        mu = self.mu_head(z)
        log_std = torch.clamp(self.logstd_head(z), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        logp = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        logp = logp.sum(-1, keepdim=True)
        return action, logp


class CoachCritic(nn.Module):
    """Q-network: (obs, action) -> scalar Q-value."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.q = mlp(obs_dim + action_dim, [256, 256], out_dim=1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)


class CoachAgent:
    def __init__(
        self,
        obs_dim=22,
        action_dim=2,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau

        self.actor = CoachActor(obs_dim, action_dim).to(self.device)

        self.critic1 = CoachCritic(obs_dim, action_dim).to(self.device)
        self.critic2 = CoachCritic(obs_dim, action_dim).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)

        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.target_critic1.eval()
        self.target_critic2.eval()

        self.actor_opt = Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.tensor([np.log(alpha)], device=self.device)

        self.alpha = self.log_alpha.exp().item()

    @torch.no_grad()
    def predict(self, obs_np, deterministic=False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if deterministic:
            mu, _ = self.actor(obs)
            action = torch.tanh(mu)
        else:
            action, _ = self.actor.sample(obs)

        return action.squeeze(0).cpu().numpy()

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])