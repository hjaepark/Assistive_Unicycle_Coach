import os
import datetime
import time
import argparse
import gymnasium as gym
import numpy as np
import random
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from envs.student import Student
from envs.coach_student_wrapper import CoachStudentWrapper
from train.monitoring.unified_monitor import UnifiedMonitor


##########################################################################################################################
##########################################################################################################################
######################## Student agent with Coach agent training code for the Unicycle Project ###########################
##########################################################################################################################
##########################################################################################################################

"""
Now using coach_student_wrapper
"""

# Basic directory structure setup
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(current_dir, "data")

# Create timestamp directory for current run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"coach_student_{timestamp}"

# Setup detailed directory structure for current run
models_dir = os.path.join(data_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model")
checkpoint_dir = os.path.join(models_dir, "checkpoints")
logs_dir = os.path.join(data_dir, "logs")

# Create all directories
directories = [
    data_dir, models_dir, best_model_dir, checkpoint_dir, logs_dir
]
for d in directories:
    os.makedirs(d, exist_ok=True)

# Maximum number of steps per episode
MAX_EPISODE_STEPS = 10000

# Success thresholds for model evaluation and saving
SUCCESS_THRESHOLDS = [1000]

# Success thresholds for model evaluation and saving (training will end after 1,000,000 total timesteps)
MAX_TIMESTEPS = 1000000

def train_coach_student(env, coach_model_path, mode: str , strategy: int, alpha: float, run_name: str):
    
    """

    Main training function
    strategy:
      1 = Continuous
      2 = Performance-Based
      3 = Progressive Disengagement
      4 = Need-Based

    """

    # Launch env at the bottom of the code

    # Apply coach+student wrapper
    wrapped_env = CoachStudentWrapper(
        env=env,
        coach_model_path=coach_model_path,
        mode=mode,
        alpha=alpha,
        
        # Mechanism to adjust student's reward, once turned off it's purely following traditional reward mechanism
        # TODO Reward mechanism will need to be reconsidered later
        reward_coef=0.01,
        reward_shaping=False
    )

    # Create evaluation environment
    # Preventing overfitting & evaluate precisely
    eval_env = gym.make('Student-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    # Apply wrapper to evaluation environment
    wrapped_eval_env = CoachStudentWrapper(
        env=eval_env,
        coach_model_path=coach_model_path,
        mode=mode,
        alpha=alpha,
        reward_coef=0.01,
        reward_shaping=False
    )

    eval_callback = EvalCallback(
        wrapped_eval_env, 
        best_model_save_path=best_model_dir,
        log_path=logs_dir, 
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Using custom logger callback from student agent training code
    ### Update monitor ###
    monitor = UnifiedMonitor(verbose=1)


    # Combine callbacks
    ### Update callback ###
    callbacks = [eval_callback, monitor]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the SAC policy
    model = SAC(
        "MlpPolicy", 
        wrapped_env, 
        verbose=1, 
        device=device, 
        tensorboard_log=logs_dir)

    start_time = time.time()
    threshold_index = 0
    total_timesteps = 0

    while threshold_index < len(SUCCESS_THRESHOLDS):
        # 1. Initialize the training
        model.learn(
            total_timesteps=10000, 
            reset_num_timesteps=False,
            tb_log_name=run_name, 
            callback=callbacks)
        total_timesteps += 10000
        
        # 2. Check the status of the training
        ### Update variable ###
        training_summary = monitor.get_training_summary()
        current_success_count = training_summary['current_success_count']
        total_episodes = training_summary['total_episodes']
        
        # 3. Update alpha based on the strategy
        new_alpha = wrapped_env.alpha

        if strategy == 1:
            # Continuous
            new_alpha = alpha # alpha remains same as input

        elif strategy == 2:
            # Performance-Based

            """
            TODO

            Test new_alpha parameters & current_success_count
            Optimize based on the result from the strategy 1
            """
            if current_success_count > 800:
                new_alpha = 0.0
            elif current_success_count > 500:
                new_alpha = 0.1
            elif current_success_count > 200:
                new_alpha = 0.3
            else:
                new_alpha = alpha

        elif strategy == 3:
            # Progressive

            """
            TODO

            Test max_steps or total steps traditional training requires
            Apply max_steps with a reasonable margin based on the result eg. if 90 -> 100
            """
            max_steps = 1_000_000
            frac = min(total_timesteps / max_steps, 1.0)
            new_alpha = alpha * (1.0 - frac)

        elif strategy == 4:
            # Need-Based
            if len(monitor.episode_successes) >= 2:
                prev_success = np.sum(monitor.episode_successes[:-1])
                current_success_count = np.sum(monitor.episode_successes)
                if current_success_count == prev_success:
                    # alpha ↑
                    new_alpha += 0.05
                else:
                    # alpha ↓
                    new_alpha -= 0.05
            
            new_alpha = max(0.0, min(0.5, new_alpha))

        # 4. Apply
        wrapped_env.set_alpha(new_alpha)
        wrapped_eval_env.set_alpha(new_alpha)

        # 5. Print training logs
        print(f"Training Progress:")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Total episodes: {training_summary['total_episodes']}")
        print(f"Current success count: {training_summary['current_success_count']}")
        print(f"Training time: {training_summary['training_time']}")
        print(f"Average reward: {training_summary['average_reward']:.2f}")
        print(f"Current alpha: {wrapped_env.alpha:.2f}") # Print alpha value
        print("-" * 50)

        # 6. Check if we've reached the success threshold
        if current_success_count >= SUCCESS_THRESHOLDS[threshold_index]:
            print(f"\nReached {SUCCESS_THRESHOLDS[threshold_index]} successes!")
            model.save(os.path.join(checkpoint_dir, f"model_{SUCCESS_THRESHOLDS[threshold_index]}_successes"))
            threshold_index += 1

        if threshold_index >= len(SUCCESS_THRESHOLDS):
            end_time = time.time()
            training_time = end_time - start_time
            print(f"\nTraining completed! All success thresholds achieved.")
            print(f"Total training time: {training_time:.2f} seconds")
            print(f"Total episodes: {total_episodes}")
            break

        # 7. Check if we've reached the maximum timesteps
        if total_timesteps >= MAX_TIMESTEPS:
            print(f"Reached maximum timesteps of {MAX_TIMESTEPS}")
            break
        
        
    
    # Save final model and plot results
    model.save(os.path.join(models_dir, f"{run_name}_final"))

def test_coach_student(env, path_to_model):
    """
    Test the trained student model WITHOUT any coach involvement
    """

    # Load the trained student model to test
    model = SAC.load(path_to_model, env=env)

    obs, info = env.reset()
    terminated = truncated = False
    step_count = 0
    total_reward = 0

    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        step_count += 1
    
    print(f"\nTest ended after {step_count} steps, total reward={total_reward}")

def check_alpha(value):
    float_value = float(value)
    if float_value < 0.0 or float_value > 1.0:
        raise argparse.ArgumentTypeError(f"{value} is invalid. Alpha must be between 0.0 and 1.0")
    return float_value

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train or test coach-student model.')
    parser.add_argument('-train', '--train', action='store_true', help='Train with coach-student wrapper')
    parser.add_argument('-test', '--test', metavar='PATH', help='Path to trained model for testing')
    parser.add_argument('--mode', type=str, default='action_merging', 
                        choices=['action_merging', 'coach_only', 'student_only'],
                        help='Action merging mode: (action_merging/coach_only/student_only)')
    parser.add_argument('--strategy', type=int, default=1, 
                        choices=[1, 2, 3, 4],
                        help='Coaching strategy: 1=Continuous, 2=Performance-Based, 3=Progressive Disengagement, 4=Need-Based')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Initial alpha value (coach intervention level, between 0.0 and 1.0)')
    parser.add_argument('-coach', '--coach', metavar='PATH', default='reference_model.zip', help='Path to coach model')
    parser.add_argument('--num_seeds', type=int, default=1,
                        help='Number of different random seeds to run the training (default: 1).')
    args = parser.parse_args()

    # Now passing strategy options too
    if args.train:
        for i in range(args.num_seeds):
            seed = i + 1

            # Fixed seed for python, numpy, and torch
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Assign seed for gym.make + reset
            env = gym.make('Student-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
            env.reset(seed=seed)

            # Assign tensorboard log
            seed_run_name = f"{run_name}_seed{seed}"
            print(f"\n[Seed={seed}] Training start => run_name: {seed_run_name}")

            # Call training function
            train_coach_student(
                env=env,
                coach_model_path=args.coach,
                mode=args.mode,
                strategy=args.strategy,
                alpha=args.alpha,
                run_name=seed_run_name
            )

    if args.test:
        if os.path.isfile(args.test):
            test_env = gym.make('Student-v0', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            test_coach_student(test_env, path_to_model=args.test)
        else:
            print(f"{args.test} not found.")