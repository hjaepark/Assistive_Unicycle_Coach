import gymnasium as gym
from stable_baselines3 import SAC
import os
import argparse
import envs
from stable_baselines3.common.callbacks import EvalCallback
import datetime
import numpy as np
from scipy.spatial.transform import Rotation
import time

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

# Success thresholds for model evaluation and saving: 10 success
SUCCESS_THRESHOLDS = [10]

# Main training function for the SAC student agent
def train(env):
    # Generate a unique timestamp for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Student_Agent_{timestamp}"

    # Create a directory to save the best models
    best_model_path = os.path.join(model_dir, f"best_{run_name}")
    os.makedirs(best_model_path, exist_ok=True)

    # Create an evaluation environment
    eval_env = gym.make('UnicyclePendulumStudent-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
   
    # Set up the evaluation callback
    eval_callback = EvalCallback(
       eval_env, 
       best_model_save_path=best_model_path,
       log_path=log_dir, 
       eval_freq=1000,
       deterministic=True, 
       render=False,
    )

    # Initialize the SAC model
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
    threshold_index = 0

    # Main training loop
    total_episodes = 0
    current_success_count = 0
    while threshold_index < len(SUCCESS_THRESHOLDS):
        # Train the model for 10000 timesteps
        model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name=run_name, callback=eval_callback)
        
        # Get the latest info from the environment
        obs, info = env.reset()
        current_success_count = info.get('success_count', current_success_count)
        total_episodes = info.get('total_episodes', total_episodes + 1)
        
        print(f"Total episodes: {total_episodes}, Total Successes: {current_success_count}")

        # Check if we've reached threshold (10 successes)
        if current_success_count >= SUCCESS_THRESHOLDS[-1]:
            end_time = time.time()
            training_time = end_time - start_time
            print(f"\nTraining completed! All success thresholds achieved.")
            print(f"Total training time: {training_time:.2f} seconds")
            print(f"Total episodes: {total_episodes}")
            break

    # Save the trained model
    model.save(os.path.join(model_dir, f"{run_name}_trained"))

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

