# train/unicycle_train/coach/coach_policy.py

import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -5, 2

def mlp(in_dim, hidden=[256,256], out_dim=0, act=nn.ReLU):
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    if out_dim > 0:
        layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

class CoachActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.body = mlp(obs_dim, [256,256])
        
        # Action heads (2D)
        self.mu_head = nn.Linear(256, action_dim)
        self.logstd_head = nn.Linear(256, action_dim)
        
        # Weight heads (1D) - NOW with mean & std!
        self.w_mu_head = nn.Linear(256, 1)
        self.w_logstd_head = nn.Linear(256, 1)

    def forward(self, obs):
        z = self.body(obs)
        
        # Action distribution
        mu = self.mu_head(z)
        log_std = torch.clamp(self.logstd_head(z), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        # Weight distribution
        w_mu = self.w_mu_head(z)
        w_logstd = torch.clamp(self.w_logstd_head(z), LOG_STD_MIN, LOG_STD_MAX)
        w_std = torch.exp(w_logstd)
        
        return mu, std, w_mu, w_std

    def sample(self, obs):
        mu, std, w_mu, w_std = self(obs)
        
        # Sample action (2D, tanh-normal)
        dist_a = torch.distributions.Normal(mu, std)
        raw_a = dist_a.rsample()
        act = torch.tanh(raw_a)
        logp_a = dist_a.log_prob(raw_a) - torch.log(1 - act.pow(2) + 1e-6)
        logp_a = logp_a.sum(-1, keepdim=True)
        
        # Sample weight (1D, logistic-normal)
        dist_w = torch.distributions.Normal(w_mu, w_std)
        raw_w = dist_w.rsample()
        w = torch.sigmoid(raw_w)
        
        # Change of variables: log |dw/d(raw_w)| = log(w*(1-w))
        logp_w = dist_w.log_prob(raw_w) - torch.log(w * (1 - w) + 1e-6)
        
        return act, logp_a, w, logp_w

class CoachPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.actor = CoachActor(obs_dim, action_dim).to(self.device)

    @torch.no_grad()
    def predict(self, obs_np, deterministic=False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if deterministic:
            mu, _, w_mu, _ = self.actor(obs)
            act = torch.tanh(mu)
            w = torch.sigmoid(w_mu)
        else:
            act, _, w, _ = self.actor.sample(obs)
        
        a = act.squeeze(0).cpu().numpy()
        weight = float(w.squeeze(0).item())
        return a, weight