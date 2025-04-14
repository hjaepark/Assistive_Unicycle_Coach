# train/unicycle_train/create_untrained_model.py

import os
import argparse
import datetime
import numpy as np
import random
import torch

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces

import envs.unicycle as unicycle
from train.unicycle_train.utils.path_utils import get_untrained_dir

def create_untrained_model(
    seed: int = 42,
    policy_kwargs: dict = None,
    learning_rate: float = 3e-4,
    buffer_size: int = 1_000_000,
    learning_starts: int = 100,
    batch_size: int = 256,
    tau: float = 0.005,
    gamma: float = 0.99,
    train_freq: int = 1,
    gradient_steps: int = 1,
    action_noise: float = None,
    replay_buffer_class: str = None,
    replay_buffer_kwargs: str = None,
    optimize_memory_usage: bool = False,
    ent_coef: str = "auto",
    target_update_interval: int = 1,
    target_entropy: str = "auto",
    use_sde: bool = False,
    sde_sample_freq: int = -1,
    use_sde_at_warmup: bool = False,
    stats_window_size: int = 100,
    tensorboard_log: str = None,
    verbose: int = 0,
    device: str = "auto",
    _init_setup_model: bool = True
):
    """
    Create an untrained model with extended observation space.
    The model is initialized with default parameters and saved to the untrained_models directory.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create and setup environment
    base_env = gym.make("Unicycle-v0")
    base_env = Monitor(base_env)
    
    # Get observation and action dimensions
    raw_obs_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]
    
    # Create model with extended observation space
    low = np.concatenate([base_env.observation_space.low,
                         np.full((action_dim,), -1.0)])
    high = np.concatenate([base_env.observation_space.high,
                          np.full((action_dim,), 1.0)])
    observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    model = SAC(
        "MlpPolicy",
        base_env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        action_noise=action_noise,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        optimize_memory_usage=optimize_memory_usage,
        ent_coef=ent_coef,
        target_update_interval=target_update_interval,
        target_entropy=target_entropy,
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        use_sde_at_warmup=use_sde_at_warmup,
        stats_window_size=stats_window_size,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        device=device,
        _init_setup_model=_init_setup_model
    )
    
    # Modify the policy's observation space
    model.policy.observation_space = observation_space

    # Save the model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(get_untrained_dir(), f"{timestamp}.zip")
    model.save(model_path)
    print(f"Created untrained model at: {model_path}")

    base_env.close()
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an untrained model with extended observation space.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Size of the replay buffer")
    parser.add_argument("--learning_starts", type=int, default=100, help="How many steps of the model to collect transitions for its learning")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size for each gradient update")
    parser.add_argument("--tau", type=float, default=0.005, help="The soft update coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor")
    parser.add_argument("--train_freq", type=int, default=1, help="Update the model every train_freq steps")
    parser.add_argument("--gradient_steps", type=int, default=1, help="How many gradient steps to do after each rollout")
    parser.add_argument("--ent_coef", type=str, default="auto", help="Entropy regularization coefficient")
    parser.add_argument("--target_update_interval", type=int, default=1, help="Update the target network every target_update_interval steps")
    parser.add_argument("--target_entropy", type=str, default="auto", help="Target entropy when learning ent_coef")
    parser.add_argument("--use_sde", action="store_true", help="Whether to use State Dependent Exploration")
    parser.add_argument("--sde_sample_freq", type=int, default=-1, help="Sample a new noise matrix every n steps when using gSDE")
    parser.add_argument("--use_sde_at_warmup", action="store_true", help="Whether to use gSDE instead of uniform sampling during the warm up phase")
    parser.add_argument("--stats_window_size", type=int, default=100, help="Window size for the rollout logging")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages")
    args = parser.parse_args()

    create_untrained_model(
        seed=args.seed,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        target_update_interval=args.target_update_interval,
        target_entropy=args.target_entropy,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        use_sde_at_warmup=args.use_sde_at_warmup,
        stats_window_size=args.stats_window_size,
        verbose=args.verbose
    )
