import os
import datetime
import time
import argparse
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import envs
from envs.student_v0 import Student
from envs.coach_student_wrapper import CoachStudentWrapper

# Using monitoring class from the student agent training code
from train.student_rl_v0 import TrainingProgressMonitor


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

# Create runs directory (top-level directory for all experiment runs)
runs_dir = os.path.join(data_dir, "runs")

# Create timestamp directory for current run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"coach_student_{timestamp}"
run_dir = os.path.join(runs_dir, f"run_{timestamp}")

# Setup detailed directory structure for current run
models_dir = os.path.join(run_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model")
checkpoint_dir = os.path.join(models_dir, "checkpoints")
plots_dir = os.path.join(run_dir, "plots")
progress_plots_dir = os.path.join(plots_dir, "progress")
final_plots_dir = os.path.join(plots_dir, "final")
logs_dir = os.path.join(run_dir, "logs")
metrics_dir = os.path.join(run_dir, "metrics")

# Create all directories
directories = [
    runs_dir, run_dir, models_dir, best_model_dir, checkpoint_dir,
    plots_dir, progress_plots_dir, final_plots_dir, logs_dir, metrics_dir
]
for d in directories:
    os.makedirs(d, exist_ok=True)

# Maximum number of steps per episode
MAX_EPISODE_STEPS = 10000

# Success thresholds for model evaluation and saving
SUCCESS_THRESHOLDS = [1000]

def train_coach_student(env, coach_model_path):
    """
    Main training function
    """
    # Apply coach+student wrapper
    wrapped_env = CoachStudentWrapper(
        env=env,
        coach_model_path=coach_model_path,
        mode="residual",
        alpha=0.5,
        reward_coef=0.01,
        reward_shaping=False
    )

    # Create evaluation environment
    eval_env = gym.make('Student-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    # Apply wrapper to evaluation environment
    wrapped_eval_env = CoachStudentWrapper(
        env=eval_env,
        coach_model_path=coach_model_path,
        mode="residual",
        alpha=0.5,
        reward_coef=0.01,      # 추가 필요
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
    training_monitor_callback = TrainingProgressMonitor(run_dir)
    # Combine callbacks
    callbacks = [eval_callback, training_monitor_callback]

    # Initialize the SAC policy
    model = SAC("MlpPolicy", wrapped_env, verbose=1, device="cuda", tensorboard_log=logs_dir)

    start_time = time.time()
    threshold_index = 0
    total_timesteps = 0

    while threshold_index < len(SUCCESS_THRESHOLDS):
        model.learn(total_timesteps=10000, reset_num_timesteps=False,
                    tb_log_name=run_name, callback=callbacks)
        total_timesteps += 10000
        
        current_success_count = training_monitor_callback.success_counts[-1] if training_monitor_callback.success_counts else 0
        total_episodes = len(training_monitor_callback.episode_counts)
        current_training_time = training_monitor_callback.training_times[-1] if training_monitor_callback.training_times else 0
        
        hours = int(current_training_time // 3600)
        minutes = int((current_training_time % 3600) // 60)
        seconds = int(current_training_time % 60)
        time_format = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"Training Progress:")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Total episodes: {total_episodes}")
        print(f"Current success count: {current_success_count}")
        print(f"Training time: {time_format}")
        print("-" * 50)

        # Check if we've reached the success threshold
        if current_success_count >= SUCCESS_THRESHOLDS[threshold_index]:
            print(f"\nReached {SUCCESS_THRESHOLDS[threshold_index]} successes!")
            # Save checkpoint model
            model.save(os.path.join(checkpoint_dir, f"model_{SUCCESS_THRESHOLDS[threshold_index]}_successes"))
            threshold_index += 1

            # If we've reached all thresholds, break the loop
            if threshold_index >= len(SUCCESS_THRESHOLDS):
                end_time = time.time()
                training_time = end_time - start_time
                print(f"\nTraining completed! All success thresholds achieved.")
                print(f"Total training time: {training_time:.2f} seconds")
                print(f"Total episodes: {total_episodes}")
                break
    
    # Save final model and plot results
    model.save(os.path.join(models_dir, f"{run_name}_final"))
    # (옵션) 최종 그래프나 결과 플롯
    # plot_training_results(training_monitor_callback, run_name, final_plots_dir)

def test_coach_student(env, path_to_model, coach_model_path):
    """
    Function to test a trained model.
    """
    wrapped_env = CoachStudentWrapper(
        env=env,
        coach_model_path=coach_model_path,
        mode="residual",
        alpha=0.5
    )

    # Load the trained model
    model = SAC.load(path_to_model, env=wrapped_env)

    obs, info = wrapped_env.reset()
    terminated = truncated = False
    step_count = 0
    total_reward = 0

    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        wrapped_env.render()
        total_reward += reward
        step_count += 1
    
    print(f"\nTest ended after {step_count} steps, total reward={total_reward}")

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train or test coach-student model.')
    parser.add_argument('-train', '--train', action='store_true', help='Train with coach-student wrapper')
    parser.add_argument('-test', '--test', metavar='PATH', help='Path to trained model for testing')
    parser.add_argument('-coach', '--coach', metavar='PATH', default='reference_model.zip', help='Path to coach model')
    args = parser.parse_args()

    # Create the unicycle environment
    env = gym.make('Student-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    if args.train:
        train_coach_student(env, coach_model_path=args.coach)

    if args.test:
        if os.path.isfile(args.test):
            test_env = gym.make('Student-v0', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            test_coach_student(test_env, path_to_model=args.test, coach_model_path=args.coach)
        else:
            print(f"{args.test} not found.")
