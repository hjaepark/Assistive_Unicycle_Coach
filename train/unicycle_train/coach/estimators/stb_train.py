# train/unicycle_train/coach/estimators/stb_train.py

import argparse
import json
import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Any

from train.unicycle_train.path import DATA_STUDENT_DIR, STB_ARTIFACTS_DIR


class Conv1D(nn.Module):
    """A configurable 1D CNN model."""
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__()
        
        # Extract model configuration
        hidden_dim = config.get('hidden_dim', 32)
        num_layers = config.get('num_layers', 1)
        kernel_size = config.get('kernel_size', 3)
        dropout_rate = config.get('dropout', 0.0)
        
        layers = []
        in_channels = input_dim
        
        # Build convolutional layers based on configuration
        for i in range(num_layers):
            # Calculate output channels for this layer
            out_channels = hidden_dim * (2 ** i) if config.get('progressive_channels', False) else hidden_dim
            
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout1d(dropout_rate))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, 1)
        
    def forward(self, x):
        # Transpose to (batch, features, seq_len) for Conv1D
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


class DataProcessor:
    """Class for handling all data processing tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.window_size = config['data']['window_size']
        self.horizon = config['data']['horizon']
        self.use_prev_reward = config['data']['use_prev_reward']
        self.feature_names = config['data']['feature_names']
        
        self.feature_scaler = StandardScaler()
        self.label_min = None
        self.label_max = None
    
    def load_csv(self, filepath: Path) -> tuple[np.ndarray, np.ndarray]:
        """Loads data from a single CSV file."""
        lines = filepath.read_text().strip().split('\n')
        
        # Find the header row, skipping any comment lines
        header_idx = 0
        for i, line in enumerate(lines):
            if not line.startswith('#'):
                header_idx = i
                break
        
        header = lines[header_idx].split(',')
        col_idx = {col.strip(): i for i, col in enumerate(header)}
        
        # Extract data rows
        data = []
        required_cols = self.feature_names + ['r']
        for line in lines[header_idx + 1:]:
            if line.strip() and not line.startswith('#'):
                values = line.split(',')
                try:
                    if all(f in col_idx for f in required_cols):
                        row = [float(values[col_idx[f]]) for f in required_cols]
                        data.append(row)
                except (KeyError, ValueError, IndexError):
                    continue
        
        if not data:
            return np.array([]), np.array([])
        
        data_np = np.array(data, dtype=np.float32)
        features = data_np[:, :len(self.feature_names)]
        rewards = data_np[:, -1]
        
        # Add previous reward as a feature if configured
        if self.use_prev_reward:
            prev_rewards = np.zeros_like(rewards)
            prev_rewards[1:] = rewards[:-1]
            prev_rewards[0] = 0  # No previous reward for the first step
            features = np.concatenate([features, prev_rewards[:, None]], axis=1)
        
        return features, rewards
    
    def create_windows(self, features: np.ndarray, rewards: np.ndarray) -> tuple:
        """Creates sliding windows from time-series data."""
        n_samples = len(features) - self.window_size - self.horizon + 1
        
        if n_samples <= 0:
            return np.array([]), np.array([])
        
        X = np.array([
            features[i : i + self.window_size] 
            for i in range(n_samples)
        ], dtype=np.float32)
        
        y = rewards[self.window_size + self.horizon - 1 :
                    self.window_size + self.horizon - 1 + n_samples]
        
        return X, y.astype(np.float32)
    
    def fit_scalers(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits scalers based on the training data."""
        n_samples, n_seq, n_features = X_train.shape
        X_flat = X_train.reshape(-1, n_features)
        self.feature_scaler.fit(X_flat)
        
        self.label_min = y_train.min()
        self.label_max = y_train.max()
    
    def transform(self, X: np.ndarray, y: np.ndarray = None) -> tuple:
        """Transforms data using the fitted scalers."""
        if len(X) == 0:
            return X, y if y is not None else None
            
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.feature_scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(original_shape).astype(np.float32)
        
        if y is not None:
            y_norm = (y - self.label_min) / (self.label_max - self.label_min + 1e-8)
            y_norm = np.clip(y_norm, 0, 1).astype(np.float32)
            return X_scaled, y_norm
        
        return X_scaled, None


def train_model(config: Dict[str, Any]):
    """Main function to train the model."""
    
    # Setup output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(
        'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' 
        else 'cpu'
    )
    
    # Initialize data processor
    processor = DataProcessor(config)
    
    # Find data directories
    data_root = Path(config['data']['data_root'])
    run_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith('run')])
    
    # Split data
    train_dirs = run_dirs[:-2] if len(run_dirs) > 2 else run_dirs
    val_dirs = run_dirs[-2:-1] if len(run_dirs) > 1 else []
    test_dirs = run_dirs[-1:] if len(run_dirs) > 0 else []
    
    # Data collection
    def collect_data(directories):
        all_X, all_y = [], []
        for dir_path in directories:
            csv_path = dir_path / 'monitor.csv'
            if csv_path.exists():
                features, rewards = processor.load_csv(csv_path)
                if len(features) > 0:
                    X, y = processor.create_windows(features, rewards)
                    if len(X) > 0:
                        all_X.append(X)
                        all_y.append(y)
        
        if all_X:
            return np.concatenate(all_X), np.concatenate(all_y)
        return np.array([]), np.array([])
    
    print(f"\nExperiment: {config['experiment_name']}")
    print(f"Using previous reward: {config['data']['use_prev_reward']}")
    print("Loading data...")
    
    X_train, y_train = collect_data(train_dirs)
    X_val, y_val = collect_data(val_dirs)
    X_test, y_test = collect_data(test_dirs)
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    if len(X_train) == 0:
        raise ValueError("No training data available!")
    
    # Preprocessing
    processor.fit_scalers(X_train, y_train)
    X_train, y_train = processor.transform(X_train, y_train)
    X_val, y_val = processor.transform(X_val, y_val) if len(X_val) > 0 else (X_val, y_val)
    X_test, y_test = processor.transform(X_test, y_test) if len(X_test) > 0 else (X_test, y_test)
    
    # Create DataLoaders
    batch_size = config['training']['batch_size']
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if len(X_val) > 0:
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[-1]
    model = Conv1D(input_dim=input_dim, config=config['model'])
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item() * len(X_batch)
            
            val_loss /= len(val_loader.dataset)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / 'best_model.pth')
            
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}")
    
    # Testing
    test_loss = None
    if len(X_test) > 0:
        if (output_dir / 'best_model.pth').exists():
            model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
        
        model.eval()
        test_loss = 0.0
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                test_loss += loss.item() * len(X_batch)
        
        test_loss /= len(test_loader.dataset)
        print(f"\nTest Loss: {test_loss:.4f}")
    
    # Save artifacts
    joblib.dump(processor.feature_scaler, output_dir / 'feature_scaler.joblib')
    
    # Save final results and config
    results = {
        **config,
        'results': {
            'input_dim': input_dim,
            'label_min': float(processor.label_min),
            'label_max': float(processor.label_max),
            'best_val_loss': float(best_val_loss) if best_val_loss != float('inf') else None,
            'test_loss': float(test_loss) if test_loss is not None else None
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel and results saved to {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D CNN Time-Series Prediction Model")
    parser.add_argument('config', type=str, help="Path to the configuration file (YAML)")
    
    args = parser.parse_args()
    
    # Load configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_model(config)

