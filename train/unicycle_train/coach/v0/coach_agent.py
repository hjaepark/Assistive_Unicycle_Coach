# train/unicycle_train/coach/v0/coach_agent.py

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

LOG_STD_MIN, LOG_STD_MAX = -5, 2

def mlp(in_dim, hidden=[256, 256], out_dim=0, act=nn.ReLU):
    """Build a simple MLP with ReLU activations."""
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    if out_dim > 0:
        layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

class CoachActor(nn.Module):
    """
    Tanh-normal policy that outputs 2D assist action.
    No weight_assist head — V0 always assists every timestep.
    """
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
        """Sample action using reparameterization trick with tanh squashing."""
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample()
        action = torch.tanh(raw)

        # Log probability with tanh correction
        logp = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        logp = logp.sum(-1, keepdim=True)

        return action, logp

class CoachCritic(nn.Module):
    """Q-network: takes (obs, action) and outputs scalar Q-value."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.q = mlp(obs_dim + action_dim, [256, 256], out_dim=1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)

class CoachAgent:
    def __init__(
        self,
        obs_dim=22,        # state(19) + student_action(2) + r_env(1)
        action_dim=2,      # 2D assist force
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        device="cuda"
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau

        # Actor
        self.actor = CoachActor(obs_dim, action_dim).to(self.device)

        # Twin critics + targets
        self.critic1 = CoachCritic(obs_dim, action_dim).to(self.device)
        self.critic2 = CoachCritic(obs_dim, action_dim).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)

        # Freeze target networks (no gradient computation)
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        # Set training mode
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.target_critic1.eval()
        self.target_critic2.eval()

        # Optimizers
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

        # Entropy temperature (single alpha)
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -float(action_dim)  # -2.0 for 2D action
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

    def train_step(self, batch):
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(
            batch["done"].astype(np.float32), dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        # Critic update
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_obs)
            q1_target = self.target_critic1(next_obs, next_action)
            q2_target = self.target_critic2(next_obs, next_action)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_logp
            backup = reward + (1 - done) * self.gamma * q_target

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        action_pi, logp_pi = self.actor.sample(obs)
        q1_pi = self.critic1(obs, action_pi)
        q2_pi = self.critic2(obs, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * logp_pi - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Temperature update
        alpha_loss = None
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft update target critics
        with torch.no_grad():
            for tp, p in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "train/critic_loss": float(critic_loss.item()),
            "train/actor_loss": float(actor_loss.item()),
            "train/q_mean": float(q_pi.mean().item()),
            "train/q1_mean": float(q1.mean().item()),
            "train/q2_mean": float(q2.mean().item()),
            "train/logp_pi": float(logp_pi.mean().item()),
            "train/action_norm": float(action_pi.norm(dim=-1).mean().item()),
            "train/alpha": float(self.alpha),
            "train/alpha_loss": float(alpha_loss.item()) if alpha_loss else 0.0,
        }

    def update_from_buffer(self, buffer, n_gradient_steps=1, batch_size=256):
        logs = {}
        for _ in range(n_gradient_steps):
            batch = buffer.sample(batch_size)
            
            # Skip if buffer is empty or too small
            if isinstance(batch, dict) and "obs" not in batch:
                break
                
            metrics = self.train_step(batch)
            for k, v in metrics.items():
                if k not in logs:
                    logs[k] = []
                logs[k].append(float(v))

        # Average all metrics across gradient steps
        logs = {k: np.mean(v) for k, v in logs.items()}

        # Append buffer statistics
        buffer_stats = buffer.get_statistics()
        logs.update(buffer_stats)

        return logs

    def save(self, path):
        """Save coach model checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
        }, path)

    def load(self, path):
        """Load coach model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])