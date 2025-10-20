# train/unicycle_train/cotraining/debug_estimator.py

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from pathlib import Path

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs.unicycle

from train.unicycle_train.coach.estimators.marb_estimator import (
    MARBConfig, MovingAverageRewardBasedEstimator
)

def test_monitor_info():
    """Test 1: Does Monitor add info["episode"]?"""
    print("=" * 60)
    print("TEST: Monitor Episode Info")
    print("=" * 60)
    
    # Create base environment
    base_env = gym.make("UnicyclePendulumPD-v0")
    
    # Wrap with Monitor
    env = Monitor(base_env)
    
    obs, info = env.reset()
    print(f"Reset successful")
    print(f"  Initial info keys: {list(info.keys())}")
    
    # Run one full episode
    done = False
    step_count = 0
    episode_reward = 0
    
    while not done and step_count < 1000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        step_count += 1
    
    print(f"\nEpisode completed: {step_count} steps, total reward: {episode_reward:.2f}")
    print(f"  Final info keys: {list(info.keys())}")
    
    # Check for episode info
    if "episode" in info:
        print(f"\nSUCCESS: info['episode'] exists!")
        print(f"  episode['r']: {info['episode']['r']:.2f}")
        print(f"  episode['l']: {info['episode']['l']}")
        print(f"  episode['t']: {info['episode']['t']:.2f}")
        return True
    else:
        print(f"\nFAIL: info['episode'] NOT found!")
        print(f"  Available keys: {list(info.keys())}")
        return False
    
    env.close()


def test_marb_estimator():
    """Test 2: Does MARB estimator work correctly?"""
    print("\n" + "=" * 60)
    print("TEST: MARB Estimator")
    print("=" * 60)
    
    # Create MARB config
    marb_config = MARBConfig(k=5, G_min=-100, G_max=100)
    estimator = MovingAverageRewardBasedEstimator(marb_config)
    
    print(f"MARB Config: k={marb_config.k}, G_min={marb_config.G_min}, G_max={marb_config.G_max}")
    
    # Test with some fake episode returns
    test_returns = [-50, -40, -30, -20, -10, 0, 10, 20, 30]
    skill_levels = []
    
    print(f"\nTesting estimator with fake returns:")
    for i, ret in enumerate(test_returns):
        skill = estimator.update(ret)
        skill_levels.append(skill)
        print(f"  Episode {i+1}: return={ret:6.1f}, skill={skill:.4f}")
    
    # Check if skill level changes
    unique_skills = len(set(skill_levels))
    if unique_skills > 1:
        print(f"\nSUCCESS: Skill level changes! ({unique_skills} unique values)")
        print(f"  Range: {min(skill_levels):.4f} -> {max(skill_levels):.4f}")
        return True
    else:
        print(f"\nFAIL: Skill level stuck at {skill_levels[0]:.4f}")
        return False


def test_monitor_with_estimator():
    """Test 3: Monitor + MARB together"""
    print("\n" + "=" * 60)
    print("TEST: Monitor + MARB Integration")
    print("=" * 60)
    
    # Setup
    base_env = gym.make("UnicyclePendulumPD-v0")
    env = Monitor(base_env)
    
    marb_config = MARBConfig(k=5, G_min=-100, G_max=100)
    estimator = MovingAverageRewardBasedEstimator(marb_config)
    
    skill_levels = []
    
    # Run 5 episodes
    for ep in range(5):
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        
        while not done and step_count < 1000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step_count += 1
        
        # Try to update estimator (like CoachWrapper does)
        if "episode" in info and "r" in info["episode"]:
            episode_return = info["episode"]["r"]
            skill_level = estimator.update(episode_return)
            skill_levels.append(skill_level)
            
            print(f"Episode {ep+1}: steps={step_count:3d}, "
                  f"return={episode_return:6.2f}, skill={skill_level:.4f}")
        else:
            print(f"Episode {ep+1}: No episode info!")
            print(f"  Available keys: {list(info.keys())}")
            skill_levels.append(0.0)
    
    env.close()
    
    # Check results
    unique_skills = len(set(skill_levels))
    if unique_skills > 1:
        print(f"\nSUCCESS: Integration works!")
        print(f"  Skill progression: {skill_levels[0]:.4f} -> {skill_levels[-1]:.4f}")
        return True
    else:
        print(f"\nFAIL: Skill level stuck at {skill_levels[0]:.4f}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ESTIMATOR DEBUG SUITE")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    try:
        # Test 1: Monitor info
        if not test_monitor_info():
            all_passed = False
            print("\nMonitor doesn't provide episode info - this is the problem!")
        
        # Test 2: MARB alone
        if not test_marb_estimator():
            all_passed = False
            print("\nMARB estimator itself is broken!")
        
        # Test 3: Integration
        if not test_monitor_with_estimator():
            all_passed = False
            print("\nIntegration between Monitor and MARB failed!")
        
        if all_passed:
            print("\n" + "=" * 60)
            print("ALL DEBUG TESTS PASSED!")
            print("=" * 60)
            print("\nNow you can use CoachWrapper safely.")
        else:
            print("\n" + "=" * 60)
            print("SOME TESTS FAILED")
            print("=" * 60)
            print("\nFix these issues before proceeding to full integration.")
            
    except Exception as e:
        print(f"\nDEBUG FAILED WITH ERROR:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()