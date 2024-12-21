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

# Create directories to hold models and logs
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from current file directory
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

##########################################################################################################################
##########################################################################################################################
############################### Student agent training code for the Unicycle Project #####################################
##########################################################################################################################
##########################################################################################################################

# Maximum number of steps per episode
MAX_EPISODE_STEPS = 10000

# Success thresholds for model evaluation and saving: 1000 success
SUCCESS_THRESHOLDS = [1000]

class TrainingMonitor(BaseCallback):
    """
    A simplified callback that monitors training progress and creates visualizations.
    Combines success counting and visualization in a more streamlined way.
    """
    def __init__(self, target_success=1000, plot_freq=5000):
        super().__init__(verbose=1)  # Enable verbose mode for basic logging
        self.target_success = target_success
        self.plot_freq = plot_freq
        
        # Training progress tracking
        self.steps = 0
        self.success_count = 0
        self.training_start = time.time()
        
        # Data collection for plotting
        self.trajectory_buffer = []  # Store recent positions for trajectory plot
        self.reward_buffer = []      # Store recent rewards
        self.buffer_size = plot_freq # Keep only recent data for plotting
        
    def _on_step(self):
        """Called after each step of training"""
        self.steps += 1
        
        # Get current state and metrics
        current_state = self.training_env.get_obs()[0]  # Get current observation
        position = current_state[:3] if current_state is not None else np.zeros(3)
        reward = self.training_env.get_reward()[0]
        self.success_count = self.training_env.get_attr('success_count')[0]
        
        # Update data buffers
        self.trajectory_buffer.append(position)
        self.reward_buffer.append(reward)
        
        # Keep buffers at fixed size
        if len(self.trajectory_buffer) > self.buffer_size:
            self.trajectory_buffer.pop(0)
            self.reward_buffer.pop(0)
        
        # Create visualization at specified intervals
        if self.steps % self.plot_freq == 0:
            self._create_training_plots()
            self._log_progress()
        
        # Check if target reached
        if self.success_count >= self.target_success:
            elapsed_time = time.time() - self.training_start
            print(f"\nTarget of {self.target_success} successes achieved!")
            print(f"Training completed in {elapsed_time:.2f} seconds")
            self._create_training_plots(final=True)
            return False
            
        return True
    
    def _create_training_plots(self, final=False):
        """Creates and saves visualization of training progress"""
        fig = plt.figure(figsize=(12, 4))
        
        # Trajectory plot
        ax1 = fig.add_subplot(121, projection='3d')
        positions = np.array(self.trajectory_buffer)
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2])
        ax1.set_title('Recent Trajectory')
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
        
        # Reward plot
        ax2 = fig.add_subplot(122)
        ax2.plot(self.reward_buffer)
        ax2.set_title('Recent Rewards')
        ax2.set_xlabel('Steps'), ax2.set_ylabel('Reward')
        
        plt.tight_layout()
        
        # Save plot
        filename = 'final_training.png' if final else f'training_step_{self.steps}.png'
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()
    
    def _log_progress(self):
        """Logs current training progress"""
        elapsed_time = time.time() - self.training_start
        print(f"\nStep {self.steps}:")
        print(f"Success count: {self.success_count}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")

# Main training function for the SAC student agent
def train(env):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Student_Agent_{timestamp}"
    
    best_model_path = os.path.join(model_dir, f"best_{run_name}")
    os.makedirs(best_model_path, exist_ok=True)
    
    eval_env = gym.make('UnicyclePendulumStudent-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    
    success_callback = SuccessCallback(target_success=1000)
    
    callbacks = CallbackList([eval_callback, success_callback])
    
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        verbose=1,
        device='cuda',
        tensorboard_log=log_dir
    )
    
    start_time = time.time()
    
    try:
        print("Starting training...")
        model.learn(
            total_timesteps=int(1e6),
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=run_name
        )
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    final_success_count = env.success_count
    print(f"\nTraining completed!")
    print(f"Final success count: {final_success_count}")
    print(f"Total training time: {training_time:.2f} seconds")
    
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
            test(test_env, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')