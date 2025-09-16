# train/unicycle_train/coach/estimators/moving_average_reward_based.py

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np

from dataclasses import dataclass

@dataclass
class Config:
    k: int # window size in episodes
    G_max: float # normalizer (global max across all runs)
    G_min: Optional[float] = None  # normalizer (global min across all runs)
    clip_low: float = 0.0
    clip_high: float = 1.0

def read_returns(path: Path) -> List[float]:
    lines = path.read_text().splitlines()
    data = [ln for ln in lines if ln and not ln.lstrip().startswith("#")]
    if not data:
        return []
    header = [h.strip() for h in data[0].split(",")]
    if "r" not in header:
        raise ValueError(f"'r' column not found in {path}")
    r_idx = header.index("r")

    out: List[float] = []
    for ln in data[1:]:
        parts = ln.split(",")
        if len(parts) > r_idx:
            try:
                out.append(float(parts[r_idx]))
            except ValueError:
                pass
    return out

def gather_runs(log_root: Path) -> Dict[str, List[float]]:
    runs: Dict[str, List[float]] = {}
    for d in sorted(log_root.iterdir()):
        if d.is_dir() and d.name.startswith("run"):
            mcsv = d / "monitor.csv"
            if mcsv.exists():
                runs[d.name] = read_returns(mcsv)
    if not runs:
        raise FileNotFoundError(f"No run*/monitor.csv under {log_root}")
    return runs

def phi(x: float, G_min: float, G_max: float) -> float:
    """Min–Max normalization to [0,1], with clipping."""
    denom = (G_max - G_min)
    if denom <= 0:
        return 0.0
    z = (x - G_min) / (denom + 1e-8)
    return float(np.clip(z, 0.0, 1.0))

def compute_mse(returns_by_run: Dict[str, List[float]],
                G_min: float,
                G_max: float,
                k: int) -> float:
    """One-step MSE on normalized values: phi(mean_k) vs phi(next)."""
    if not (G_max > G_min):
        return float("inf")
    
    errs: List[float] = []
    for R in returns_by_run.values():
        if len(R) < k + 1:
            continue
        # t indexes the last idx of the k-window. target is t+1
        for t in range(k - 1, len(R) - 1):
            window = R[t - k + 1: t + 1]
            G_bar = float(np.mean(window)) # moving average over last k
            target = R[t + 1] # next episode's return
            pred = phi(G_bar, G_min, G_max) # predictor: phi(mean)
            targ = phi(target, G_min, G_max) # target: phi(next)
            diff = pred - targ
            errs.append(diff * diff)

    return float(np.mean(errs)) if errs else float("inf")

def precompute(log_root: Path, k_candidates: Sequence[int] = range(5, 11)) -> Tuple[Config, dict]:
    """
    Determine (k, G_min, G_max) from multiple runs under logs_root.

    - Load episode returns 'r' for each run.
    - Set G_min/max from all observed returns.
    - For each k in k_candidates, compute MSE(k).
    - Pick the k with the smallest MSE.
    """
    rb = gather_runs(log_root)
    all_r = [x for R in rb.values() for x in R]
    if not all_r:
        raise ValueError("No episode returns found.")
    G_min = float(np.min(all_r))
    G_max = float(np.max(all_r))

    best_k, best_mse, table = None, float("inf"), []
    for k in k_candidates:
        mse = compute_mse(rb, G_min, G_max, k)
        table.append({"k": int(k), "mse": float(mse)})
        if mse < best_mse:
            best_k, best_mse = int(k), mse

    if best_k is None:
        raise RuntimeError("Not enough episodes for any k.")

    cfg = Config(k=best_k, G_max=G_max, G_min=G_min)
    info = {"G_min": G_min, "G_max": G_max, "runs": len(rb), "mse_table": table}
    return cfg, info


class MovingAverageRewardBasedEstimator:
    """
    Compute estimation during runtime.
    theta^(t) = clip( phi( (1/k_t) * sum_{i=t-k_t+1}^{t} r_i ), clip_low, clip_high )
    """
    def __init__(self, cfg: Config):
        self.k = int(cfg.k)
        self.G_max = float(cfg.G_max)
        self.G_min = float(cfg.G_min) if cfg.G_min is not None else None
        self.clip_low = cfg.clip_low
        self.clip_high = cfg.clip_high
        self.reset()

    def reset(self):
        self.buf = []
        self.sum = 0.0
        self.t = 0

    def update(self, G_t: float) -> float:
        self.t += 1
        self.buf.append(G_t); self.sum += G_t
        if len(self.buf) > self.k:
            self.sum -= self.buf.pop(0)
        k_t = self.t if self.t < self.k else self.k
        avg = self.sum / k_t if k_t > 0 else 0.0

        if self.G_min is not None and (self.G_max > self.G_min):
            s = phi(avg, self.G_min, self.G_max)
        elif self.G_max > 0:
            s = float(np.clip(avg / self.G_max, 0.0, 1.0))
        else:
            s = 0.0
        s = float(np.clip(s, self.clip_low, self.clip_high))
        return s