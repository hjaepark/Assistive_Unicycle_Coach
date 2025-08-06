# train/unicycle_train/coach/trajectory_based_skill_estimation.py
# RLlib 2.12.0 – legacy ("old") stack

"""
Train a “coach” SAC agent that linearly blends the student’s action with a fixed expert policy.  
The **blend‑weight w** is produced online by a trajectory‑based episode estimator hat e(T).

"""

import datetime
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Dict, Deque, DefaultDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import ray
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.evaluation.episode import Episode

from train.unicycle_train import path as P

E_MAX: int = 600                  # avg episodes for solo mastery
WIN_STEPS: int = 50               # trajectory window length
GRU_HIDDEN: int = 64              # hidden size of GRU
LR: float = 1e-3                  # learning rate for f_θ
INIT_W: float = 1.0               # start with 100 % coach assist
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────
# (1) environment factory – shared by student & coach
# ────────────────────────────────────────────────────────────────

def unicycle_env_creator(env_config: Dict[str, Any]):
    """Gymnasium factory registered for RLlib."""
    import gymnasium as gym, envs.unicycle
    return gym.make("UnicyclePendulumPD-v0", **env_config)


tune.register_env("UnicycleStudentRLlib", unicycle_env_creator)

tune.register_env(
    "UnicycleCoachStudentRLlib",
    lambda cfg: __import__(
        "train.unicycle_train.coach.action_blending_wrapper",
        fromlist=["ActionBlendingWrapper"],
    ).ActionBlendingWrapper(cfg),

)
# ────────────────────────────────────────────────────────────────
# (2)  callback factory – trajectory‑based online skill estimation
# ────────────────────────────────────────────────────────────────

def SkillEstimationCallbackFactory(e_max: int = E_MAX, win: int = WIN_STEPS):
    """Returns a subclass of DefaultCallbacks that maintains *w* via f_θ."""

    # ――――――――――――――――――――――――――――――――――――――――――――――――――――――
    # small GRU regressor   (17‑dim obs × 50  →  scalar ê)
    # ――――――――――――――――――――――――――――――――――――――――――――――――――――――
    class _Estimator(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=17, hidden_size=GRU_HIDDEN, batch_first=True)
            self.head = nn.Linear(GRU_HIDDEN, 1)

        def forward(self, x):             # x: (B, win, 17)
            y, _ = self.gru(x)
            return self.head(y[:, -1]).squeeze(-1)

    # ――――――――――――――――――――――――――――――――――――――――――――――――――――――
    class _SkillCB(DefaultCallbacks):
        """Internal RLlib callback used by the coach experiment."""

        def __init__(self):
            super().__init__()
            self.model = _Estimator().to(DEVICE)
            self.opt = optim.Adam(self.model.parameters(), lr=LR)
            self.loss_fn = nn.MSELoss()
            self.buffers: DefaultDict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=win))
            self.e_max = float(e_max)

        # --------------------------------------------------
        # RLlib hooks
        # --------------------------------------------------
        def on_episode_start(self, *, episode: Episode, **__):
            self.buffers[episode.env_id].clear()

        def on_episode_step(self, *, base_env, episode: Episode, env_index: int, **__):
            env_id = episode.env_id
            env = base_env.get_sub_environments()[env_index]
            obs = env.unwrapped._get_obs()
            if obs is None:
                return
            self.buffers[env_id].append(np.asarray(obs, np.float32))
            if len(self.buffers[env_id]) < win:
                return  # not enough for a prediction yet

            # 1) ê̂ prediction ------------------------------------------------
            traj = np.stack(self.buffers[env_id])[None]          # (1, win, 17)
            with torch.no_grad():
                e_hat = float(self.model(torch.from_numpy(traj).to(DEVICE)).cpu().item())

            # 2) weight computation ----------------------------------------
            w_new = float(np.clip(1.0 - e_hat / self.e_max, 0.0, 1.0))
            for env in base_env.get_sub_environments():
                if hasattr(env, "set_weight"):
                    env.set_weight(w_new)

            # 3) one‑step supervised update --------------------------------
            info = episode.last_info_for()
            if info is None:
                return
            true_e = torch.tensor([float(info.get("num_episodes", 0))], device=DEVICE)
            pred = self.model(torch.from_numpy(traj).to(DEVICE))
            loss = self.loss_fn(pred, true_e)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # 4) logging ---------------------------------------------------
            episode.custom_metrics.update(w=w_new, e_hat=e_hat, skill_loss=float(loss.item()))

        def on_train_result(self, *, result: dict, **__):
            it = result["training_iteration"]
            rew = result["episode_reward_mean"]
            wcur = result["custom_metrics"].get("w_mean", np.nan)
            ehat = result["custom_metrics"].get("e_hat_mean", np.nan)
            loss = result["custom_metrics"].get("skill_loss_mean", np.nan)
            print(
                f"[it={it:>4}]  reward={rew:8.2f}  w̄={wcur:5.3f}  ê̄={ehat:6.1f}  skill‑loss={loss:7.4f}"
            )

    return _SkillCB

# ────────────────────────────────────────────────────────────────
# (3)  main launcher
# ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"✓ trajectory‑based skill estimator | E_MAX={E_MAX}")

    env_conf = {
        "env_cfg": {},
        "expert_ckpt": P.EXPERT_MODEL_DIR,
        "initial_w": INIT_W,
    }

    sac_cfg = (
        SACConfig()
        .environment("UnicycleCoachStudentRLlib", env_config=env_conf)
        .framework("torch")
        .resources(num_gpus=1)
        .callbacks(SkillEstimationCallbackFactory())
    ).to_dict()

    logs_root = P.LOGS_COACH_DIR
    run_name = f"student_trajectory_based_skill_estimation_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    tuner = tune.Tuner(
        "SAC",
        param_space=sac_cfg,
        run_config=RunConfig(
            name=run_name,
            storage_path=str(logs_root),
            stop={"timesteps_total": 500_000},
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=20,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max",
                num_to_keep=2,
            ),
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    ray.shutdown()
    main()
