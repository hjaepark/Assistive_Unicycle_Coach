# train/unicycle_train/coach/coach_config.py

from dataclasses import dataclass
import numpy as np

@dataclass
class CoachRewardConfig:
    """Coach reward function configuration"""
    
    # Component toggles
    use_safety: bool
    use_improvement: bool
    use_energy_penalty: bool
    use_usage_penalty: bool
    use_responsibility: bool
    
    # Hyperparameters
    c_safe: float
    beta_u: float
    beta_p: float
    lambda_E: float
    lambda_W0: float
    rho: float
    p: float
    c_fall: float
    b_goal: float
    resp_ema_alpha: float
    kappa: np.ndarray