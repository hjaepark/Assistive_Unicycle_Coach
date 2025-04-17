# train/unicycle_train/create_trained_model.py

import os
import argparse
import datetime
import numpy as np
import random
import torch

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

import envs.unicycle as unicycle
from train.unicycle_train.utils.path_utils import (
    get_models_dir,
    get_logs_dir,
    get_train_logs_dir
)
from train.unicycle_train.wrapper.extended_observation_wrapper import ExtendedObservationWrapper
from train.unicycle_train.monitor_callback.student_monitor import StudentMonitor

class TrainingStopCallback(BaseCallback):
    """
    Callback to stop training when reaching target success count.
    """
    def __init__(
        self,
        target_success_count: int = 1000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.target_success_count = target_success_count
        self.last_episode = -1

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            current_episode = info.get("num_episodes", -1)
            
            # Print only at the start of a new episode
            if current_episode != self.last_episode:
                self.last_episode = current_episode
                obs = self.locals["new_obs"]
                print("\n" + "="*50)
                print(f"Episode {current_episode} Observation Check")
                print("="*50)
                print(f"Total shape: {obs.shape}")
                print("\nOriginal observation (first 15 values):")
                for i, val in enumerate(obs[0, :15]):
                    print(f"  obs[{i:2d}] = {val:10.6f}")
                print("\nExtended observation (last 4 values):")
                for i, val in enumerate(obs[0, 15:]):
                    print(f"  obs[{i+15:2d}] = {val:10.6f}")
                print("\nExtended part all zeros:", np.all(obs[0, 15:] == 0))
                print("="*50 + "\n")

            if "num_success" in info:
                current_success = info["num_success"]
                self.logger.record("train/num_success", current_success)
                
                if current_success >= self.target_success_count:
                    if self.verbose > 0:
                        print(f"[TrainingStopCallback] Reached target success count: {current_success}")
                    return False
        return True

def create_trained_model(
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
    verbose: int = 0,
    device: str = "cuda",
    _init_setup_model: bool = True,
    total_timesteps: int = 10**7,
    target_success_count: int = 1000
):
    """
    Create and train a model with extended observation space.
    The model is trained and saved to the final_models directory.
    The best model during training is saved to the best_models directory.
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create and setup environment
    base_env = gym.make("Unicycle-v0")
    base_env = Monitor(base_env)
    env = ExtendedObservationWrapper(base_env)
    
    # Create evaluation environment
    eval_env = gym.make("Unicycle-v0")
    eval_env = Monitor(eval_env)
    eval_env = ExtendedObservationWrapper(eval_env)
    
    # Create directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"unicycle_student_{timestamp}"
    
    models_dir = get_models_dir()
    best_model_dir = os.path.join(models_dir, "best_model", run_name)
    final_model_dir = os.path.join(models_dir, "final_model")
    logs_dir = os.path.join(get_train_logs_dir(), run_name)
    
    # Create directories if not exist
    directories = [models_dir, best_model_dir, final_model_dir, logs_dir]
    for d in directories:
        os.makedirs(d, exist_ok=True)
    
    # Create model
    model = SAC(
        "MlpPolicy",
        env,
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
        verbose=verbose,
        device=device,
        _init_setup_model=_init_setup_model,
        tensorboard_log=logs_dir
    )
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=logs_dir,
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    success_stop_callback = TrainingStopCallback(
        target_success_count=target_success_count,
        verbose=verbose
    )
    
    student_monitor = StudentMonitor()
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, success_stop_callback, student_monitor],
        progress_bar=True
    )
    
    # Save the final model
    final_model_path = os.path.join(final_model_dir, f"{timestamp}.zip")
    model.save(final_model_path)
    print(f"Saved final model at: {final_model_path}")
    
    env.close()
    eval_env.close()
    return final_model_path

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create and train a model with extended observation space.")
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
    parser.add_argument("--total_timesteps", type=int, default=10**7, help="Total number of timesteps to train for")
    parser.add_argument("--target_success_count", type=int, default=1000, help="Target number of successes to reach before stopping")
    args = parser.parse_args()

    # Create and train model with command line arguments
    create_trained_model(
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
        verbose=args.verbose,
        total_timesteps=args.total_timesteps,
        target_success_count=args.target_success_count
    )