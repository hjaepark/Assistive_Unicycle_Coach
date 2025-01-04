import gymnasium as gym
from stable_baselines3 import SAC
import os
import argparse
import envs
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import datetime
import numpy as np
from scipy.spatial.transform import Rotation
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


##########################################################################################################################
##########################################################################################################################
############################### Student agent training code for the Unicycle Project #####################################
##########################################################################################################################
##########################################################################################################################


# Basic directory structure setup
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(current_dir, "data")

# Create runs directory (top-level directory for all experiment runs)
runs_dir = os.path.join(data_dir, "runs")

# Create timestamp directory for current run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"student_{timestamp}"
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

# Additional directories for long-term storage
archive_dir = os.path.join(data_dir, "archive")
shared_dir = os.path.join(data_dir, "shared")

# Create all directories
directories = [
    runs_dir, run_dir, models_dir, best_model_dir, checkpoint_dir,
    plots_dir, progress_plots_dir, final_plots_dir, logs_dir, 
    metrics_dir, archive_dir, shared_dir
]
for dir_path in directories:
    os.makedirs(dir_path, exist_ok=True)

# Maximum number of steps per episode
MAX_EPISODE_STEPS = 10000

# Success thresholds for model evaluation and saving
SUCCESS_THRESHOLDS = [1000]

class TrainingProgressMonitor(BaseCallback):
    """
    Custom callback for logging unicycle positions, actions, and success information during training.
    """
    def __init__(self, run_dir, verbose=0):
        super(TrainingProgressMonitor, self).__init__(verbose)
        # Initialize directory paths for organized data storage
        self.run_dir = run_dir
        self.progress_plots_dir = os.path.join(run_dir, "plots", "progress")
        self.metrics_dir = os.path.join(run_dir, "metrics")

        # Essential tracking variables
        self.unicycle_positions = []
        self.actions = []
        self.timesteps = []
        self.success_counts = []
        self.success_rates = []
        self.avg_goal_times = []
        
        # Episode tracking
        self.episode_counts = []
        self.episode_rewards = []
        self.current_episode = 0
        self.current_episode_reward = 0
        
        # Time tracking
        self.training_times = []
        self.start_time = time.time()
        
        # Performance monitoring
        self.timestep_rewards = []
        self.roll_angles = []
        self.pitch_angles = []
        self.yaw_angles = []
        self.wheel_velocities = []
        self.steering_angles = []

    def _on_step(self):
        """
        This method is called at each step of the training process.
        It logs various metrics for later analysis and plotting.
        """
        obs = self.locals['new_obs'][0]
        info = self.locals['infos'][0]
        action = self.locals['actions'][0]

        # Update current eposide reward
        self.current_episode_reward += info['reward']
        
        # Store basic tracking infos
        self.unicycle_positions.append(obs[:3])
        self.actions.append(action)
        self.timesteps.append(self.num_timesteps)
        self.success_counts.append(info['success_count'])
        
        # Update urrent timestep reward
        self.timestep_rewards.append(info['reward'])

        # Extract and store angle infos
        quat = obs[3:7]
        euler = Rotation.from_quat(np.roll(quat, -1)).as_euler('xyz')
        self.roll_angles.append(euler[0])
        self.pitch_angles.append(euler[1])
        self.yaw_angles.append(euler[2])     

        # Store control infos
        self.wheel_velocities.append(obs[21])
        self.steering_angles.append(action[1])

        # Calculate success rates
        recent_successes = sum([1 for s in self.success_counts[-100:] if s > 0])
        self.success_rates.append(recent_successes / min(100, len(self.success_counts)))   
        
        # Calculate goal reaching time
        if info['goal_reached']:
            self.avg_goal_times.append(info['steps'])
        else:
            self.avg_goal_times.append(self.avg_goal_times[-1] if self.avg_goal_times else 0)

        # Store training time
        self.training_times.append(time.time() - self.start_time)

        # Episode ends & plot graph every 1000 episodes
        if info.get('terminal_observation') is not None:
            self.current_episode += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_counts.append(self.current_episode)
            self.current_episode_reward = 0
            
            if self.current_episode % 1000 == 0:
                self.plot_progress()

        return True

    def plot_progress(self):
        """
        Enhanced visualization method with 6 subplots showing different aspects of training
        """
        fig = plt.figure(figsize=(20, 30))
        
        # Plot 1: Reward per Timestep (top left)
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(self.timesteps, self.timestep_rewards)
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('Reward per Timestep')
        ax1.grid(True)
        
        # Plot 2: Reward per Episode (top right)
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(self.episode_counts, self.episode_rewards)
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Episode Reward')
        ax2.set_title('Reward per Episode')
        ax2.grid(True)
        
        # Plot 3: 3D Trajectory (middle left)
        ax3 = fig.add_subplot(3, 2, 3, projection='3d')
        positions = np.array(self.unicycle_positions)
        ax3.plot3D(positions[:, 0], positions[:, 1], positions[:, 2])
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Unicycle 3D Trajectory')
        
        # Plot 4: Control Inputs (middle right)
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(self.timesteps, self.wheel_velocities, label='Wheel Velocity')
        ax4.plot(self.timesteps, self.steering_angles, label='Steering Angle')
        ax4.set_xlabel('Timesteps')
        ax4.set_ylabel('Control Input')
        ax4.set_title('Control Inputs Over Time')
        ax4.legend()
        ax4.grid(True)
        
        # Plot 5: Angular Changes (bottom left)
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(self.timesteps, self.roll_angles, label='Roll')
        ax5.plot(self.timesteps, self.pitch_angles, label='Pitch')
        ax5.plot(self.timesteps, self.yaw_angles, label='Yaw')
        ax5.set_xlabel('Timesteps')
        ax5.set_ylabel('Angles (rad)')
        ax5.set_title('Angular Changes')
        ax5.legend()
        ax5.grid(True)
        
        # Plot 6: Action Distribution (bottom right)
        ax6 = fig.add_subplot(3, 2, 6)
        actions = np.array(self.actions)
        ax6.hist2d(actions[:, 0], actions[:, 1], bins=30)
        ax6.set_xlabel('Action 1')
        ax6.set_ylabel('Action 2')
        ax6.set_title('Action Distribution')
        
        plt.tight_layout()
        plot_path = os.path.join(self.progress_plots_dir, f'progress_{self.current_episode}.png')
        plt.savefig(plot_path)
        plt.close()
    
def plot_training_results(logger, run_name, final_plots_dir):
    try:
        num_episodes = len(logger.episode_counts)
        aligned_success_rates = logger.success_rates[-num_episodes:]
        aligned_goal_times = logger.avg_goal_times[-num_episodes:]
        
        episode_success_counts = []
        current_count = 0
        for count in logger.success_counts:
            if count > current_count:
                current_count = count
            episode_success_counts.append(current_count)
        aligned_success_counts = episode_success_counts[-num_episodes:]

        fig = plt.figure(figsize=(20, 20))
        
        # Plot 1: Success Rate Progress (top left)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(logger.episode_counts, aligned_success_rates)
        ax1.set_xlabel('Total Episodes')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Learning Progress')
        ax1.grid(True)

        # Plot 2: 3D Trajectory (top right)
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        positions = np.array(logger.unicycle_positions[-10000:])
        ax2.plot3D(positions[:, 0], positions[:, 1], positions[:, 2])
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Final Episode Trajectory')

        # Plot 3: Goal Reach Time (bottom left)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(aligned_success_counts, aligned_goal_times)
        ax3.set_xlabel('Cumulative Successes')
        ax3.set_ylabel('Average Goal Reach Time (steps)')
        ax3.set_title('Task Efficiency')
        ax3.grid(True)
        ax3.axvline(x=1000, color='g', linestyle='--', alpha=0.5,
                    label='Target Success Count')
        ax3.legend()

        # Plot 4: Angular Changes (bottom right)
        ax4 = fig.add_subplot(2, 2, 4)
        recent_timesteps = logger.timesteps[-1000:]
        ax4.plot(recent_timesteps, logger.roll_angles[-1000:], label='Roll')
        ax4.plot(recent_timesteps, logger.pitch_angles[-1000:], label='Pitch')
        ax4.set_xlabel('Timesteps')
        ax4.set_ylabel('Angles (rad)')
        ax4.set_title('Final Episode Stability')
        ax4.legend()
        ax4.grid(True)

        plt.suptitle(f'Training Results for {run_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(final_plots_dir, 'training_results.png'))
        
    except Exception as e:
        print(f"Error during plotting final results: {e}")
        
    finally:
        plt.close('all')

def train(env):
    """
    Main training function for the student agent.
    Implements proper success tracking and termination conditions.
    """
    # Create an evaluation environment
    eval_env = gym.make('Student-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    
    # Set up the evaluation callback
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=best_model_dir,
        log_path=logs_dir, 
        eval_freq=10000,
        deterministic=True, 
        render=False,
    )

    # Set up the custom logger callback
    unicycle_logger = TrainingProgressMonitor(run_dir)
    
    # Combine callbacks
    callbacks = [eval_callback, unicycle_logger]

    # Initialize the SAC policy student model
    model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=logs_dir)

    start_time = time.time()
    threshold_index = 0
    total_timesteps = 0
    
    # Main training loop with proper success tracking
    while threshold_index < len(SUCCESS_THRESHOLDS):
        # Train the model for 10000 timesteps
        model.learn(total_timesteps=10000, reset_num_timesteps=False, 
                   tb_log_name=run_name, callback=callbacks)
        total_timesteps += 10000
        
        # Get monitoring metrics from the logger
        current_success_count = unicycle_logger.success_counts[-1] if unicycle_logger.success_counts else 0
        total_episodes = len(unicycle_logger.episode_counts)
        current_training_time = unicycle_logger.training_times[-1] if unicycle_logger.training_times else 0

        hours = int(current_training_time // 3600)
        minutes = int((current_training_time % 3600) // 60)
        seconds = int(current_training_time % 60)
        time_format = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Print comprehensive training progress
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
    plot_training_results(unicycle_logger, run_name, final_plots_dir)

def test(env, path_to_model):
    """
    Function to test a trained model.
    Test does not need following requirements. Only the basic test bench needed to render the model.
    """
    
    # Load the trained model
    model = SAC.load(path_to_model, env=env)

    obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0
    step_count = 0
    
    # Main testing loop
    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        # Get the model's prediction and take a step in the environment
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        total_reward += reward
        step_count += 1
        
        # Extract relevant information from the observation
        unicycle_pos = obs[:3]
        unicycle_quat = obs[3:7]
        pendulum_quat = obs[7:11]
        
        # Convert quaternions to Euler angles
        unicycle_euler = Rotation.from_quat(np.roll(unicycle_quat, -1)).as_euler('xyz', degrees=False)
        pendulum_euler = Rotation.from_quat(np.roll(pendulum_quat, -1)).as_euler('xyz', degrees=False)
        
        wheel_velocity = obs[21]  # Assuming wheel velocity is at index 21

        # Print step information
        print(f"Step: {step_count}, Reward: {reward:.4f}")
        print(f"Unicycle Position: ({unicycle_pos[0]:.4f}, {unicycle_pos[1]:.4f}, {unicycle_pos[2]:.4f})")
        print(f"Unicycle Roll, Pitch, Yaw: ({unicycle_euler[0]:.4f}, {unicycle_euler[1]:.4f}, {unicycle_euler[2]:.4f})")
        print(f"Pendulum Roll, Pitch, Yaw: ({pendulum_euler[0]:.4f}, {pendulum_euler[1]:.4f}, {pendulum_euler[2]:.4f})")
        print(f"Wheel Velocity: {wheel_velocity:.4f}")
        print(f"Success Count: {info['success_count']}")
        print("------------------------------")

    # Print final episode statistics
    print(f"\nEpisode finished after {step_count} steps")
    print(f"Total reward: {total_reward}")
    print(f"Average reward per step: {total_reward / step_count}")
    print(f"Total successes: {info['success_count']}")
    
    if info.get('goal_reached', False):
        print("Episode ended by reaching the goal!")
    elif terminated:
        print("Episode ended by termination condition.")
    elif truncated:
        print("Episode ended by truncation (max steps reached).")

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train or test student model.')
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('-test', '--test', metavar='path_to_model')
    args = parser.parse_args()

    # Create the unicycle environment
    env = gym.make('Student-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    if args.train:
        train(env)

    if args.test:
        if os.path.isfile(args.test):
            test_env = gym.make('Student-v0', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            test(test_env, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')