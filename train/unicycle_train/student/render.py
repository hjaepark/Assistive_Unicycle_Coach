# train/unicycle_train/student/render.py
import sys

import time
import ray
from ray.rllib.algorithms.sac import SAC

def unicycle_env_creator(env_config):
    import gymnasium as gym
    import envs.unicycle
    env_config = env_config or {}
    env_config["render_mode"] = "human"  # Enable rendering
    return gym.make("UnicyclePendulumPD-v0", **env_config)

def main():
    if len(sys.argv) < 2:
        print("python render.py <checkpoint_path>")
        return
    
    checkpoint_path = sys.argv[1]
    
    # Initialize Ray and register environment
    ray.init()
    from ray import tune
    tune.register_env("UnicycleStudentRLlib", unicycle_env_creator)
    
    # Load algorithm
    algo = SAC(config={
        "env": "UnicycleStudentRLlib",
        "env_config": {
            "reset_noise_scale": 0.01,
            "kp": 40.0,
            "kd": 6.0,
        },
        "framework": "torch",
        "num_env_runners": 0,
    })
    algo.restore(checkpoint_path)
    
    # Create environment and run episodes
    env = unicycle_env_creator({"reset_noise_scale": 0.01, "kp": 40.0, "kd": 6.0})
    
    episode = 0
    print("Starting infinite rendering (Press Ctrl+C to stop)")
    
    try:
        while True:
            episode += 1
            obs, _ = env.reset()
            print(f"\nEpisode: {episode}")
            
            step = 0
            while True:
                step += 1
                action = algo.compute_single_action(obs, explore=False)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                time.sleep(0.01)
                
                if terminated or truncated:
                    print(f"  Complete: {'Success' if info.get('goal_reached') else 'Fail'} ({step} steps)")
                    break
                    
    except KeyboardInterrupt:
        print("\n\nRendering stopped")
    finally:
        env.close()
        ray.shutdown()

if __name__ == "__main__":
    main()