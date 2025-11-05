# train/unicycle_train/cotraining/config_loader.py

import yaml
import numpy as np

def load_config(config_path: str) -> dict:
    # Load YAML config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, save_path: str):
    # Save config to YAML file
    def convert_numpy(obj):
        # Convert numpy arrays to lists
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    config_clean = convert_numpy(config)
    
    with open(save_path, 'w') as f:
        yaml.dump(config_clean, f, default_flow_style=False, sort_keys=False)