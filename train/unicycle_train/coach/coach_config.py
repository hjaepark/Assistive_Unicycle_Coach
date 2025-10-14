# train/unicycle_train/coach/coach_config.py

from dataclasses import dataclass
import numpy as np

@dataclass
class CoachRewardConfig:
    """Coach reward function configuration"""
    
    # Main component weights
    use_safety: bool = True
    c_safe: float = 0.5
    
    use_improvement: bool = True
    beta_u: float = 2.0 # upright improvement weight
    beta_p: float = 0.05 # progress improvement weight

    use_energy_penalty: bool = True
    lambda_E: float = 0.5 # energy penalty coefficient

    use_usage_penalty: bool = True
    lambda_W0: float = 0.02 # base usage penalty
    rho: float = 2.0 # skill scaling exponent
    p: float = 1.0 # usage penalty power

    use_responsibility: bool = True
    c_fall: float = 10.0 # fall penalty
    b_goal: float = 5.0 # goal bonus
    
    # EMA for responsibility tracking
    resp_ema_alpha: float = 0.1
    
    # Kappa (assistance scaling)
    kappa: np.ndarray = None
    
    def __post_init__(self):
        if self.kappa is None:
            self.kappa = np.array([0.3, 0.3], dtype=np.float32)