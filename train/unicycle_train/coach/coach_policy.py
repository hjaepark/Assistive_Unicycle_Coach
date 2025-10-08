# train/unicycle_train/coach/coach_policy.py

import torch, torch.nn as nn

# Gaussian policy log std limits for numerical stability
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
    # Action distribution (mu, log_std) + weight (sigmoid)
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.body = mlp(obs_dim, [256,256])
        self.mu_head = nn.Linear(256, action_dim)
        self.logstd_head = nn.Linear(256, action_dim)
        self.weight_head = nn.Linear(256, 1)

    def forward(self, obs):
        z = self.body(obs)
        mu = self.mu_head(z)
        log_std = torch.clamp(self.logstd_head(z), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        weight_assist = torch.sigmoid(self.weight_head(z))  # [B,1]
        return mu, std, weight_assist

    def sample(self, obs):
        mu, std, weight_assist = self(obs)
        dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample()
        act = torch.tanh(raw)
        logp = dist.log_prob(raw) - torch.log(1 - act.pow(2) + 1e-6)
        logp = logp.sum(-1, keepdim=True)
        return act, logp, weight_assist

class CoachPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.actor = CoachActor(obs_dim, action_dim).to(self.device)

    @torch.no_grad()
    def predict(self, obs_np, deterministic=False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0) # [1, obs_dim]

        mu, _, weight_assist = self.actor(obs)
        if deterministic:
            act = torch.tanh(mu)
        else:
            act, _, weight_assist = self.actor.sample(obs)
        
        a = act.squeeze(0).cpu().numpy()
        w = float(weight_assist.squeeze(0).item())
        return a, w
