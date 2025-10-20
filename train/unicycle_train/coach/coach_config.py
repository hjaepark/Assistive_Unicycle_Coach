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
    beta_u: float = 0.5      # 2.0 → 0.5
    beta_p: float = 0.01     # 0.05 → 0.01

    use_energy_penalty: bool = True
    lambda_E: float = 2.0    # 0.5 → 2.0

    use_usage_penalty: bool = True
    lambda_W0: float = 0.2   # 0.02 → 0.2
    rho: float = 2.0
    p: float = 2.0           # 1.0 → 2.0

    use_responsibility: bool = True
    c_fall: float = 10.0
    b_goal: float = 5.0
    
    # EMA for responsibility tracking
    resp_ema_alpha: float = 0.1
    
    # Kappa (assistance scaling)
    kappa: np.ndarray = None
    
    def __post_init__(self):
        if self.kappa is None:
            self.kappa = np.array([0.3, 0.3], dtype=np.float32)