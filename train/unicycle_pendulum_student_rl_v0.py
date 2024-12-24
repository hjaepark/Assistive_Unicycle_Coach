import gymnasium as gym
from stable_baselines3 import SAC
import os
import argparse
import envs
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
import datetime
import numpy as np
from scipy.spatial.transform import Rotation
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Create directories to hold models and logs
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(current_dir, "data")
model_dir = os.path.join(data_dir, "models")
log_dir = os.path.join(data_dir, "logs")
csv_dir = os.path.join(data_dir, "csv")
plot_dir = os.path.join(data_dir, "plot")

# Create directories
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Maximum number of steps per episode
MAX_EPISODE_STEPS = 10000

# Success thresholds for model evaluation and saving
SUCCESS_THRESHOLDS = [1000]

class TrainingMonitor(BaseCallback):
    """
    Training progress monitoring and visualization callback.
    Tracks success rate, trajectory, rewards, and creates periodic visualizations.
    """
    def __init__(self, target_success=1000, plot_freq=10000):
        super().__init__(verbose=1)
        self.target_success = target_success
        self.plot_freq = plot_freq
        
        # Training data storage
        self.all_positions = []          # Store unicycle positions
        self.unicycle_roll_pitch_yaw = [] # Store unicycle orientation
        self.pendulum_roll_pitch_yaw = [] # Store pendulum orientation
        self.all_rewards = []            # Store rewards
        self.all_success_counts = []     # Store cumulative successes
        self.all_timestamps = []         # Store timing information
        self.success_trajectories = []   # Store successful trajectories
        self.current_trajectory = []     # Store current episode trajectory
        self.wheel_velocities = []       # Store wheel velocities
        
        # Training progress tracking
        self.stacked_rewards = []
        self.stacked_success_rates = []
        self.stacked_positions = []
        
        self.steps = 0
        self.success_count = 0
        self.training_start = time.time()
        self.last_success_count = 0
        
    def _on_step(self):
        """Process each training step and collect data"""
        self.steps += 1
        
        # Obtain currnet status and environment 
        # self.locals "dictionary" stores current status
        obs = self.locals['new_obs'][0]
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        
        position = obs[:3]
        unicycle_quat = obs[3:7]
        pendulum_quat = obs[7:11]
        wheel_velocity = obs[21]
        
        unicycle_euler = Rotation.from_quat(np.roll(unicycle_quat, -1)).as_euler('xyz')
        pendulum_euler = Rotation.from_quat(np.roll(pendulum_quat, -1)).as_euler('xyz')
        
        # Store in list
        self.all_positions.append(position)
        self.current_trajectory.append(position)
        self.unicycle_roll_pitch_yaw.append(unicycle_euler)
        self.pendulum_roll_pitch_yaw.append(pendulum_euler)
        self.wheel_velocities.append(wheel_velocity)
        self.all_rewards.append(reward)
        self.success_count = info['success_count']
        self.all_success_counts.append(self.success_count)
        self.all_timestamps.append(time.time() - self.training_start)
        
        # Store trajectory once reached goal
        if info.get('goal_reached', False):
            self.success_trajectories.append(np.array(self.current_trajectory))
            self.current_trajectory = []
        
        if self.steps % self.plot_freq == 0:
            # Average reward calculation
            current_rewards = self.all_rewards[-self.plot_freq:]
            avg_reward = np.mean(current_rewards) if current_rewards else 0
            self.stacked_rewards.append(avg_reward)
            
            # Success rate calculation
            success_in_interval = self.success_count - self.last_success_count
            success_rate = success_in_interval / self.plot_freq
            self.stacked_success_rates.append(success_rate)
            
            # Store current position
            self.stacked_positions.append(np.array(self.all_positions[-self.plot_freq:]))
            
            self.last_success_count = self.success_count

            self._create_training_plots()
            self._log_progress()
            
        return True
    
    def _create_training_plots(self):
        """Create a comprehensive visualization showing training progress across intervals"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Average reward per interval
        ax1 = fig.add_subplot(221)
        intervals = np.arange(len(self.stacked_rewards)) * self.plot_freq
        ax1.plot(intervals, self.stacked_rewards, marker='o', color='blue')
        ax1.set_title('Average Reward per Interval')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Average Reward')
 
        if len(self.stacked_rewards) > 5:
            moving_avg = pd.Series(self.stacked_rewards).rolling(window=5).mean()
            ax1.plot(intervals, moving_avg, color='red', linestyle='--', label='Moving Average')
        ax1.grid(True)
        ax1.legend()
        
        # 2. Success rate per interval
        ax2 = fig.add_subplot(222)
        ax2.plot(intervals, self.stacked_success_rates, marker='o', color='green')
        ax2.set_title('Success Rate per Interval')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Success Rate')
        ax2.grid(True)
        
        # 3. Recent trajectory
        ax3 = fig.add_subplot(223, projection='3d')
        recent_positions = self.stacked_positions[-1] if self.stacked_positions else None
        if recent_positions is not None:
            ax3.plot(recent_positions[:, 0], recent_positions[:, 1], recent_positions[:, 2], 
                    color='blue', alpha=0.6, label='Recent Trajectory')
            ax3.set_title(f'Recent Trajectory (Steps {self.steps-self.plot_freq} to {self.steps})')
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Y Position')
        ax3.set_zlabel('Z Position')
        ax3.grid(True)
        ax3.legend()
        
        # 4. Culmulative average success rate
        ax4 = fig.add_subplot(224)
        cumulative_success_rate = np.cumsum(self.stacked_success_rates) / (np.arange(len(self.stacked_success_rates)) + 1)
        ax4.plot(intervals, cumulative_success_rate, marker='o', color='purple')
        ax4.set_title('Cumulative Average Success Rate')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Success Rate')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'training_progress_{self.steps}.png'))
        plt.close()
    
    def _log_progress(self):
        """Log current training progress"""
        elapsed_time = time.time() - self.training_start
        print(f"\nTraining Progress - Step {self.steps}:")
        print(f"Success count: {self.success_count}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Success rate: {self.success_count/self.steps:.4f}")
    
    def _save_final_data(self):
        """Save all training data to CSV files for later analysis"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save position and orientation data
        position_orientation_df = pd.DataFrame({
            'step': range(len(self.all_positions)),
            'x': [p[0] for p in self.all_positions],
            'y': [p[1] for p in self.all_positions],
            'z': [p[2] for p in self.all_positions],
            'unicycle_roll': [rpy[0] for rpy in self.unicycle_roll_pitch_yaw],
            'unicycle_pitch': [rpy[1] for rpy in self.unicycle_roll_pitch_yaw],
            'unicycle_yaw': [rpy[2] for rpy in self.unicycle_roll_pitch_yaw],
            'pendulum_roll': [rpy[0] for rpy in self.pendulum_roll_pitch_yaw],
            'pendulum_pitch': [rpy[1] for rpy in self.pendulum_roll_pitch_yaw],
            'pendulum_yaw': [rpy[2] for rpy in self.pendulum_roll_pitch_yaw],
            'wheel_velocity': self.wheel_velocities
        })
        position_orientation_df.to_csv(os.path.join(csv_dir, f'motion_data_{timestamp}.csv'), index=False)
        
        # Save training metrics
        training_metrics_df = pd.DataFrame({
            'step': range(len(self.all_rewards)),
            'reward': self.all_rewards,
            'success_count': self.all_success_counts,
            'time': self.all_timestamps
        })
        training_metrics_df.to_csv(os.path.join(csv_dir, f'training_metrics_{timestamp}.csv'), index=False)

def train(env):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Student_Agent_{timestamp}"
    
    # Initialize model
    model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    # Set up monitoring
    monitor = TrainingMonitor(target_success=1000, plot_freq=5000)
    
    # Start training
    try:
        print("Starting training...")

        while True:
            model.learn(
                total_timesteps=int(1e5),  
                callback=monitor,
                tb_log_name=run_name,
                reset_num_timesteps=False
            )
            
            # Check if reached target number
            if monitor.success_count >= monitor.target_success:
                elapsed_time = time.time() - monitor.training_start
                print(f"\nTarget of {monitor.target_success} successes achieved!")
                print(f"Training completed in {elapsed_time:.2f} seconds")
                monitor._create_training_plots()
                monitor._save_final_data()
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        monitor._create_training_plots()
        monitor._save_final_data()
    
    # Save final model
    final_model_path = os.path.join(model_dir, f"{run_name}_final")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

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
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train or test SAC model.')
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('-test', '--test', metavar='path_to_model')
    args = parser.parse_args()

    # Create the unicycle environment
    env = gym.make('UnicyclePendulumStudent-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    if args.train:
        train(env)

    if args.test:
        if os.path.isfile(args.test):
            test_env = gym.make('UnicyclePendulumStudent-v0', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            
            import glfw
            if glfw.init():
                glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
                glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            
            obs, info = test_env.reset()
            
            test_env.render()

            window = glfw.get_current_context()
            if window:
                glfw.set_window_size(window, 1280, 720)
            
            test(test_env, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')

