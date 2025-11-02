# train/unicycle_train/coach/coach_config.py

from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class CoachRewardConfig:
    """
    Coach reward function configuration

    LP_t = ψ(w_t) · r_t^shaped
    where ψ(w_t) = (1 - min{w_t/ρ, 1})^2
    
    r_coach = LP_t - β*w_t - λ_E*||a^C||^2
    
    Parameters:
        rho: assisted weight threshold for ψ(w_t)
        beta: usage penalty weight
        lambda_E: energy penalty weight
        kappa: intervention scaling factor [kappa_x, kappa_y]
    """
    
    rho: float = 0.5
    beta: float = 0.05
    lambda_E: float = 0.1
    kappa: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.kappa is None:
            self.kappa = np.array([0.3, 0.3], dtype=np.float32)
        elif not isinstance(self.kappa, np.ndarray):
            self.kappa = np.array(self.kappa, dtype=np.float32)