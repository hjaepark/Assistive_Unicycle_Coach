# train/unicycle_train/coach/estimators/stb_estimator.py

import json
import joblib
import torch
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional

# Import the Conv1D model class from stb_train
from .stb_train import Conv1D


@dataclass
class STBConfig:
    """Configuration for STB estimator"""
    artifacts_dir: Path
    device: str = 'cpu'
    n: Optional[int] = None  # Will be loaded from config
    d: Optional[int] = None  # Will be loaded from config


class StateTrajectoryBasedEstimator:
    """
    Loads pre-trained STB model and artifacts to perform
    online (real-time) skill estimation.
    """
    def __init__(self, config: STBConfig):
        print(f"[STB Estimator] Loading artifacts from: {config.artifacts_dir}")
        self.device = torch.device(config.device)
        self.artifacts_dir = config.artifacts_dir

        # 1. Load results and configuration file
        results_path = config.artifacts_dir / 'results.json'
        with open(results_path, 'r') as f:
            self.full_config = json.load(f)

        model_config = self.full_config['model']
        data_config = self.full_config['data']
        results_config = self.full_config['results']

        self.window_size = data_config['window_size']
        self.use_prev_reward = data_config['use_prev_reward']
        self.feature_names = data_config['feature_names']
        
        # Set n and d for compatibility with train_estimator.py
        self.n = self.window_size
        self.d = len(self.feature_names) + (1 if self.use_prev_reward else 0)
        
        # 2. Load trained model (.pth)
        model_path = config.artifacts_dir / 'best_model.pth'
        input_dim = results_config['input_dim']
        
        # Recreate model structure based on config
        self.model = Conv1D(input_dim=input_dim, config=model_config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Set to inference mode
        print(f"[STB Estimator] Model loaded successfully (window_size={self.n}, feature_dim={self.d})")

        # 3. Load feature scaler (.joblib)
        scaler_path = config.artifacts_dir / 'feature_scaler.joblib'
        self.scaler = joblib.load(scaler_path)
        print("[STB Estimator] Scaler loaded successfully")

        # 4. Initialize history buffer for real-time data
        self.history = deque(maxlen=self.window_size)
        self.prev_reward = 0.0  # For tracking previous reward if needed
        self.reset()

    def reset(self):
        """Reset history when episode ends"""
        self.history.clear()
        self.prev_reward = 0.0

    def update(self, observation: np.ndarray, reward: Optional[float] = None):
        """
        Add new observation to history.
        
        Args:
            observation: Current observation vector
            reward: Current reward (optional, used if use_prev_reward is True)
        """
        feature = observation.astype(np.float32)
        
        # Add previous reward as feature if configured
        if self.use_prev_reward:
            feature = np.concatenate([feature, [self.prev_reward]])
            if reward is not None:
                self.prev_reward = reward
        
        self.history.append(feature)

    def current(self) -> float:
        """
        Predict skill based on current history.
        Returns normalized skill value in [0, 1].
        """
        # Need full window size of data for prediction
        if len(self.history) < self.window_size:
            return 0.0  # Return default value if insufficient data

        # 1. Prepare input data (shape: [1, window_size, num_features])
        X = np.stack(list(self.history))[None, :, :]
        
        # 2. Feature scaling
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(original_shape)

        # 3. Perform prediction with model
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
            prediction_normalized = self.model(X_tensor).item()

        # 4. Clip to [0, 1] range and return
        return np.clip(prediction_normalized, 0.0, 1.0)