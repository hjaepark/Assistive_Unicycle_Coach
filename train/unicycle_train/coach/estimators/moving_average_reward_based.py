# train/unicycle_train/coach/estimators/moving_average_reward_based.py

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np

from dataclasses import dataclass

@dataclass
class Config:
    k: int # window size in episodes
    G_max: float # normalizer (global max across all runs)
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

def compute_mse(returns_by_run: Dict[str, List[float]], G_max: float, k: int) -> float:
    if not (G_max > 0):
        return float("inf")

    errs: List[float] = []
    for R in returns_by_run.values():
        if len(R) < k + 1:
            continue 
        for t in range(k - 1, len(R) - 1):
            window = R[max(0, t - k + 1): t + 1]
            G_bar = float(np.mean(window)) # moving average over last k
            target = R[t + 1] # next episode's return
            diff = (G_bar / G_max) - (target / G_max)
            errs.append(diff * diff)

    return float(np.mean(errs)) if errs else float("inf")

def precompute(log_root: Path, k_candidates: Sequence[int] = range(5, 11)) -> Tuple[Config, dict]:
    """
    Determine (k, G_max) from multiple runs under logs_root.

    - Load episode returns 'r' for each run.
    - Set G_max := max of all observed returns (global max across runs).
    - For each k in k_candidates, compute MSE(k).
    - Pick the k with the smallest MSE.
    """
    rb = gather_runs(log_root)
    all_r = [x for R in rb.values() for x in R]
    if not all_r:
        raise ValueError("No episode returns found.")
    G_max = max(all_r)

    best_k, best_mse, table = None, float("inf"), []
    for k in k_candidates:
        mse = compute_mse(rb, G_max, k)
        table.append({"k": int(k), "mse": float(mse)})
        if mse < best_mse:
            best_k, best_mse = int(k), mse

    if best_k is None:
        raise RuntimeError("Not enough episodes for any k.")

    cfg = Config(k=best_k, G_max=G_max)
    info = {"G_max": G_max, "runs": len(rb), "mse_table": table}
    return cfg, info


class MovingAverageRewardBasedEstimator:
    """
    Compute estimation during runtime.
    theta^(t) = clip( (1/k) * sum_{i=t-k+1}^{t} r_i / G_max, clip_low, clip_high )
    """
    def __init__(self, cfg: Config):
        self.k = int(cfg.k)
        self.G_max = float(cfg.G_max)
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
        s = (avg / self.G_max) if self.G_max > 0 else 0.0
        return max(self.clip_low, min(self.clip_high, s))