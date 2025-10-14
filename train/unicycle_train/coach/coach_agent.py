# train/unicycle_train/coach/coach_agent.py

import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from train.unicycle_train.coach.coach_policy import CoachPolicy, mlp

# Q-function approximation network
class CoachCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # obs + action + weight_assist(1-dim) input
        self.q = mlp(obs_dim + action_dim + 1, [256, 256], out_dim=1)
    
    def forward(self, obs, act, weight_assist):
        x = torch.cat([obs, act, weight_assist], dim=-1)
        return self.q(x)


class CoachAgent:
    def __init__(self, obs_dim=22, action_dim=2, lr=3e-4, gamma=0.99, 
                 tau=0.005, alpha=0.2, auto_alpha=True, device="cuda"):
        self.device = torch.device(device)

        # Policy and Actor
        self.policy = CoachPolicy(obs_dim, action_dim, device=device)
        self.actor = self.policy.actor

        # Critics and Target Critics
        self.critic1 = CoachCritic(obs_dim, action_dim).to(self.device)
        self.critic2 = CoachCritic(obs_dim, action_dim).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)

        # Freeze target critics (no gradients) & set evaluation mode
        # Ensure target networks are not updated by optimizer
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False
        self.critic1.train(); self.critic2.train()
        self.target_critic1.eval(); self.target_critic2.eval()
        self.actor.train()

        # Optimizers
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)

        self.gamma = gamma
        self.tau = tau
        
        # Temperature (entropy coefficient) auto-tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            # Target entropy heuristic: -dim(A)
            # We only count the action dimension, not weight_assist
            # because weight is sigmoid-bounded and has naturally lower entropy
            self.target_entropy = -float(action_dim) # -2.0
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.tensor([np.log(alpha)], device=self.device)

        self.alpha = self.log_alpha.exp().item()

    @torch.no_grad()
    def predict(self, obs_np, deterministic=False):
        return self.policy.predict(obs_np, deterministic=deterministic)

    def train_step(self, batch):
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"].astype(np.float32), dtype=torch.float32, device=self.device).unsqueeze(-1)
        weight_assist = torch.as_tensor(batch["weight_assist"], dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Compute target Q-value
        with torch.no_grad():
            a2, logp2, g2 = self.actor.sample(next_obs)
            q1_t = self.target_critic1(next_obs, a2, g2)
            q2_t = self.target_critic2(next_obs, a2, g2)
            q_tgt = torch.min(q1_t, q2_t) - self.alpha * logp2
            backup = reward + (1 - done) * self.gamma * q_tgt

        # Update critics
        q1 = self.critic1(obs, action, weight_assist)
        q2 = self.critic2(obs, action, weight_assist)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # Update actor
        a_pi, logp_pi, g_pi = self.actor.sample(obs)
        q1_pi = self.critic1(obs, a_pi, g_pi)
        q2_pi = self.critic2(obs, a_pi, g_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp_pi - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Update temperature (entropy coefficient)
        alpha_loss = None
        if self.auto_alpha:
            # Entropy should be close to target_entropy
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft update target critics (polyak averaging)
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
            "train/weight_assist_mean": float(g_pi.mean().item()),
            "train/action_norm": float(a_pi.norm(dim=-1).mean().item()),
            "train/alpha": float(self.alpha),
            "train/alpha_loss": float(alpha_loss.item()) if alpha_loss else 0.0,
        }

    def update_from_buffer(self, buffer, n_gradient_steps=10, batch_size=256):
        logs = {}
        for _ in range(n_gradient_steps):
            batch = buffer.sample(batch_size)
            out = self.train_step(batch)
            for k, v in out.items():
                if k not in logs:
                    logs[k] = []
                logs[k].append(float(v))
        
        # Average over all gradient steps
        logs = {k: np.mean(v) for k, v in logs.items()}
        
        # Add buffer statistics
        buffer_stats = buffer.get_statistics()
        logs.update(buffer_stats)
        
        return logs