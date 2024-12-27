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

class UnicyclePositionLogger(BaseCallback):
    """
    Custom callback for logging unicycle positions, actions, and success information during training.
    """
    def __init__(self, run_dir, verbose=0):
        super(UnicyclePositionLogger, self).__init__(verbose)
        # Initialize directory paths for organized data storage
        self.run_dir = run_dir
        self.progress_plots_dir = os.path.join(run_dir, "plots", "progress")
        self.metrics_dir = os.path.join(run_dir, "metrics")

        self.unicycle_positions = []
        self.actions = []
        self.timesteps = []
        self.success_timesteps = []
        self.success_counts = []
        self.total_episodes = []
        self.success_rates = []
        self.avg_rewards = []
        self.avg_goal_times = []
        self.training_times = []
        self.start_time = time.time()

        # Progress update variables
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_counts = []
        self.current_episode = 0
        self.current_episode_reward = 0

    def _on_step(self):
        """
        This method is called at each step of the training process.
        It logs various metrics for later analysis and plotting.
        """
        obs = self.locals['new_obs'][0]
        info = self.locals['infos'][0]
        self.current_episode_reward += info['reward']
        
        
        self.unicycle_positions.append(obs[:3])  # Log unicycle position
        self.actions.append(self.locals['actions'][0])  # Log action taken
        self.timesteps.append(self.num_timesteps)  # Log current timestep
        
        self.success_counts.append(info['success_count'])
        self.total_episodes.append(info['total_episodes'])
        
        # Calculate success rate over last 100 episodes
        recent_successes = sum([1 for s in self.success_counts[-100:] if s > 0])
        self.success_rates.append(recent_successes / min(100, len(self.success_counts)))
        
        # Calculate average reward over last 100 episodes
        self.avg_rewards.append(np.mean([info['reward'] for info in self.locals['infos'][-100:]]))
        
        # Calculate average goal reach time
        if info['goal_reached']:
            self.avg_goal_times.append(info['steps'])
        else:
            self.avg_goal_times.append(self.avg_goal_times[-1] if self.avg_goal_times else 0)
        
        self.training_times.append(time.time() - self.start_time)

        # Plot progress every 1000 episodes
        if info.get('terminal_observation') is not None:
            self.current_episode += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_successes.append(info['success_count'])
            self.episode_counts.append(self.current_episode)
            self.current_episode_reward = 0
            
            if self.current_episode % 1000 == 0:
                self.plot_progress()
        
        return True
    
    def plot_progress(self):
        """
        Updated plotting function to save files in the new directory structure
        """
        # Plot and save success graph
        plt.figure(figsize=(12, 8))
        plt.plot(self.episode_counts, self.episode_successes, 'b.-')
        plt.title('Cumulative Successes over Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Successes')
        plt.grid(True)
        plt.savefig(os.path.join(self.progress_plots_dir, f'progress_{self.current_episode}.png'))
        plt.close()

        # Plot and save rewards graph
        plt.figure(figsize=(12, 8))
        plt.plot(self.episode_counts, self.episode_rewards, 'r.-')
        plt.title('Episode Rewards over Time')
        plt.xlabel('Episodes')
        plt.ylabel('Episode Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.progress_plots_dir, f'rewards_{self.current_episode}.png'))
        plt.close()

        # Save metrics to CSV
        metrics_data = {
            'episode': self.episode_counts,
            'success_count': self.episode_successes,
            'reward': self.episode_rewards,
            'success_rate': self.success_rates,
            'avg_reward': self.avg_rewards,
            'avg_goal_time': self.avg_goal_times
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(self.metrics_dir, f'metrics_{self.current_episode}.csv'), 
                         index=False)

def plot_training_results(logger, run_name):
    """
    Function to plot various training metrics.
    """
    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    fig.suptitle(f'Training Results for {run_name}', fontsize=16)

    # 1. Success rate vs episodes graph
    axs[0, 0].plot(logger.total_episodes, logger.success_rates)
    axs[0, 0].set_xlabel('Total Episodes')
    axs[0, 0].set_ylabel('Success Rate')
    axs[0, 0].set_title('Success Rate vs Episodes')

    # 2. Average reward vs episodes graph
    axs[0, 1].plot(logger.total_episodes, logger.avg_rewards)
    axs[0, 1].set_xlabel('Total Episodes')
    axs[0, 1].set_ylabel('Average Reward')
    axs[0, 1].set_title('Average Reward vs Episodes')

    # 3. Goal reach time vs success count graph
    axs[1, 0].plot(logger.success_counts, logger.avg_goal_times)
    axs[1, 0].set_xlabel('Cumulative Successes')
    axs[1, 0].set_ylabel('Average Goal Reach Time (steps)')
    axs[1, 0].set_title('Goal Reach Time vs Successes')

    # 4. Training time vs success count graph
    axs[1, 1].plot(logger.success_counts, logger.training_times)
    axs[1, 1].set_xlabel('Cumulative Successes')
    axs[1, 1].set_ylabel('Total Training Time (seconds)')
    axs[1, 1].set_title('Training Time vs Successes')

    # 5. Performance metrics comparison bar graph
    thresholds = ['10^2', '10^3', '10^4']
    metrics = ['Avg Reward', 'Avg Goal Time', 'Success Rate']
    data = np.array([
        [logger.avg_rewards[i] for i in [100, 1000, 10000]],
        [logger.avg_goal_times[i] for i in [100, 1000, 10000]],
        [logger.success_rates[i] for i in [100, 1000, 10000]]
    ])

    x = np.arange(len(thresholds))
    width = 0.25

    for i in range(len(metrics)):
        axs[2, 0].bar(x + i*width, data[i], width, label=metrics[i])

    axs[2, 0].set_xlabel('Success Thresholds')
    axs[2, 0].set_ylabel('Performance Metrics')
    axs[2, 0].set_title('Performance Metrics Comparison')
    axs[2, 0].set_xticks(x + width)
    axs[2, 0].set_xticklabels(thresholds)
    axs[2, 0].legend()

    # Highlighting 10^2, 10^3, 10^4 points in relevant graphs
    for ax in [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]:
        for threshold in [100, 1000, 10000]:
            ax.axvline(x=threshold, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(final_plots_dir, 'training_results.png'))
    plt.close()

def train(env):
    """
    Main training function for the SAC agent.
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
    unicycle_logger = UnicyclePositionLogger(run_dir)
    
    # Combine callbacks
    callbacks = [eval_callback, unicycle_logger]

    # Initialize the SAC model
    model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=logs_dir)

    start_time = time.time()
    threshold_index = 0

    # Main training loop
    total_episodes = 0
    current_success_count = 0
    while threshold_index < len(SUCCESS_THRESHOLDS):
        # Train the model for 10000 timesteps
        model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name=run_name, callback=callbacks)
        
        # Get the latest info from the environment
        obs, info = env.reset()
        current_success_count = info.get('success_count', current_success_count)
        total_episodes = info.get('total_episodes', total_episodes + 1)
        
        print(f"Total episodes: {total_episodes}, Total Successes: {current_success_count}")
        
        # Check if we've reached the 100 episodes
        if total_episodes % 100 == 0:
            unicycle_logger.plot_progress()

        """
        # Check if we've reached the next success threshold
        if current_success_count >= SUCCESS_THRESHOLDS[threshold_index]:
            print(f"\nReached {SUCCESS_THRESHOLDS[threshold_index]} successes!")
            model.save(f"{best_model_path}/model_{SUCCESS_THRESHOLDS[threshold_index]}_successes")
            threshold_index += 1
        """

        # Check if we've reached threshold (1000)
        # 1000 success MAX
        if current_success_count >= SUCCESS_THRESHOLDS[-1]:
            end_time = time.time()
            training_time = end_time - start_time
            print(f"\nTraining completed! All success thresholds achieved.")
            print(f"Total training time: {training_time:.2f} seconds")
            print(f"Total episodes: {total_episodes}")
            break

    # Update model saving path
    model.save(os.path.join(models_dir, f"{run_name}_final"))
    
    # Update final results plotting
    plot_training_results(unicycle_logger, run_name, final_plots_dir)

def test(env, path_to_model):
    """
    Function to test a trained model.
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
    # Set up command line argument parsing₩
    parser = argparse.ArgumentParser(description='Train or test SAC model.')
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