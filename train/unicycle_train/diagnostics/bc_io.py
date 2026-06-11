# train/unicycle_train/diagnostics/bc_io.py
from pathlib import Path
import numpy as np
import torch
from train.unicycle_train.coach.v0.coach_agent import CoachActor

V1_LOGS = Path(__file__).resolve().parents[1] / "coach" / "v1" / "logs"


def load_bc_actor(device, checkpoint=None):
    if checkpoint is None:
        runs = sorted(V1_LOGS.glob("bc_*"))
        if not runs:
            raise FileNotFoundError(f"no bc_* runs in {V1_LOGS}")
        checkpoint = runs[-1] / "checkpoints" / "best_bc_actor.pth"
    ckpt = Path(checkpoint)
    actor = CoachActor(obs_dim=22, action_dim=2).to(device)
    actor.load_state_dict(torch.load(ckpt, map_location=device))
    actor.eval()
    for p in actor.parameters():
        p.requires_grad = False
    return actor, ckpt


@torch.no_grad()
def predict_delta(actor, obs, action, reward, device):
    # Input layout must match build_dataset.py: state(19) + action(2) + reward(1)
    x = np.concatenate([obs, action, [reward]]).astype(np.float32)
    mu, _ = actor(torch.as_tensor(x, device=device).unsqueeze(0))
    return torch.tanh(mu).squeeze(0).cpu().numpy()