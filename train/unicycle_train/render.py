import os
import argparse
import gymnasium as gym
from stable_baselines3 import SAC
import envs.unicycle as unicycle
from train.unicycle_train.wrapper.extended_observation_wrapper import ExtendedObservationWrapper
from train.unicycle_train.utils.path_utils import get_models_dir

def render_model(model_path: str):
    """
    Load and render a trained model infinitely until user interruption.
    
    Args:
        model_path: Path to the trained model
    """
    # Create environment with rendering
    base_env = gym.make("Unicycle-v0", render_mode="human")
    env = ExtendedObservationWrapper(base_env)
    
    # Load the model
    model = SAC.load(model_path)
    
    # Run episodes infinitely
    episode = 0
    try:
        while True:
            episode += 1
            print(f"\nStarting episode {episode}")
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Check if goal is reached and end episode immediately
                if info.get("goal_reached", False):
                    print(f"Goal reached at step {steps}")
                    break
                
                if done or truncated:
                    print(f"Episode {episode} finished after {steps} steps")
                    print(f"Total reward: {total_reward:.2f}")
                    if "goal_reached" in info:
                        print(f"Goal reached: {info['goal_reached']}")
    
    except KeyboardInterrupt:
        print("\nRendering stopped by user")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    args = parser.parse_args()
    
    render_model(args.model_path) 