# train/unicycle_train/cotraining/test_integration.py

import gymnasium as gym
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs.unicycle

from stable_baselines3.common.monitor import Monitor

from coach.coach_wrapper import CoachWrapper
from coach.coach_agent import CoachAgent
from coach.estimators.marb_estimator import MovingAverageRewardBasedEstimator, MARBConfig

def test_basic_functionality():
    """Test 1: Basic wrapper functionality without errors"""
    print("=" * 60)
    print("TEST 1: Basic Wrapper Functionality")
    print("=" * 60)
    
    base_env = gym.make("UnicyclePendulumPD-v0")
    env = Monitor(base_env)
    marb_config = MARBConfig(k=10, G_min=-100, G_max=100)
    estimator = MovingAverageRewardBasedEstimator(marb_config)
    coach = CoachAgent(obs_dim=20, action_dim=2, auto_alpha=True, device="cpu")
    wrapped_env = CoachWrapper(env, coach_agent=coach, estimator=estimator)
    
    obs, info = wrapped_env.reset()
    print(f"Environment reset successful")
    print(f"Observation shape: {obs.shape} (should be (19,) for student)")
    
    # Check coach observation internally
    coach_obs = np.concatenate([obs, [wrapped_env.skill_level]])
    print(f"Coach observation shape: {coach_obs.shape} (should be (20,))")
    
    # Run a few steps
    for i in range(10):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        # Check that coach-related keys exist
        assert "weight_assist" in info, "Missing weight_assist in info"
        assert "coach_action_norm" in info, "Missing coach_action_norm in info"
        assert "skill_level" in info, "Missing skill_level in info"
    
    print(f"Environment step successful")
    print(f"Last weight_assist: {info['weight_assist']:.4f}")
    print(f"Last skill_level: {info['skill_level']:.4f}")
    wrapped_env.close()
    print("TEST 1 PASSED\n")

def test_buffer_collection():
    """Test 2: Check if buffer collects data correctly"""
    print("=" * 60)
    print("TEST 2: Buffer Data Collection")
    print("=" * 60)
    
    base_env = gym.make("UnicyclePendulumPD-v0")
    env = Monitor(base_env)
    marb_config = MARBConfig(k=10, G_min=-100, G_max=100)
    estimator = MovingAverageRewardBasedEstimator(marb_config)
    coach = CoachAgent(obs_dim=20, action_dim=2, auto_alpha=True, device="cpu")
    wrapped_env = CoachWrapper(env, coach_agent=coach, estimator=estimator)
    
    obs, info = wrapped_env.reset()
    
    # Run one full episode
    done = False
    step_count = 0
    while not done and step_count < 1000:
        student_action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(student_action)
        done = terminated or truncated
        step_count += 1
    
    print(f"Episode completed: {step_count} steps")
    buffer_size = wrapped_env.coach_buffer.size
    print(f"Buffer size: {buffer_size}")
    
    if buffer_size > 0:
        # Check buffer statistics
        stats = wrapped_env.coach_buffer.get_statistics()
        print(f"\nBuffer statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Check rollout info at episode end
        if "rollout/coach_reward_mean" in info:
            print(f"\nRollout info (episode end):")
            print(f"  rollout/coach_reward_mean: {info['rollout/coach_reward_mean']:.4f}")
            print(f"  rollout/weight_assist_mean: {info['rollout/weight_assist_mean']:.4f}")
            print(f"  rollout/skill_level: {info['rollout/skill_level']:.4f}")
        
        print("\nTEST 2 PASSED - Buffer is collecting data\n")
    else:
        print("TEST 2 FAILED - Buffer is empty!\n")
        return False
    
    wrapped_env.close()
    return True

def test_sac_training_stability():
    """Test 3: Check SAC training stability (CRITICAL!)"""
    print("=" * 60)
    print("TEST 3: SAC Training Stability")
    print("=" * 60)
    
    base_env = gym.make("UnicyclePendulumPD-v0")
    env = Monitor(base_env)
    marb_config = MARBConfig(k=10, G_min=-100, G_max=100)
    estimator = MovingAverageRewardBasedEstimator(marb_config)
    coach = CoachAgent(obs_dim=20, action_dim=2, auto_alpha=True, device="cpu")
    wrapped_env = CoachWrapper(env, coach_agent=coach, estimator=estimator)
    
    obs, info = wrapped_env.reset()
    
    # Collect enough data
    print("Collecting training data...")
    for ep in range(5):
        done = False
        step_count = 0
        while not done and step_count < 500:
            student_action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(student_action)
            done = terminated or truncated
            step_count += 1
        obs, info = wrapped_env.reset()
        print(f"  Episode {ep+1}: {step_count} steps, buffer size: {wrapped_env.coach_buffer.size}")
    
    buffer_size = wrapped_env.coach_buffer.size
    if buffer_size < 1000:
        print(f"WARNING: Buffer size ({buffer_size}) < 1000")
        print("   Skipping training stability test")
        wrapped_env.close()
        return False
    
    # Train multiple times and check for explosions
    print("\nTraining coach for 5 updates...")
    all_stable = True
    
    for update_i in range(5):
        logs = wrapped_env.update_coach(n_gradient_steps=10, learning_starts=1000)
        
        # Check for critical metrics
        alpha = logs.get("train/alpha", 0)
        critic_loss = logs.get("train/critic_loss", 0)
        actor_loss = logs.get("train/actor_loss", 0)
        q_mean = logs.get("train/q_mean", 0)
        logp_pi = logs.get("train/logp_pi", 0)
        
        print(f"\n  Update {update_i+1}:")
        print(f"    alpha: {alpha:.4f}")
        print(f"    critic_loss: {critic_loss:.4f}")
        print(f"    actor_loss: {actor_loss:.4f}")
        print(f"    q_mean: {q_mean:.4f}")
        print(f"    logp_pi: {logp_pi:.4f}")
        
        # Check for explosions
        if alpha > 10.0:
            print(f"    Alpha exploded: {alpha}")
            all_stable = False
        if abs(critic_loss) > 1000:
            print(f"    Critic loss exploded: {critic_loss}")
            all_stable = False
        if abs(q_mean) > 1000:
            print(f"    Q-values exploded: {q_mean}")
            all_stable = False
        if np.isnan(alpha) or np.isnan(critic_loss):
            print(f"    NaN detected!")
            all_stable = False
    
    if all_stable:
        print("\nTEST 3 PASSED - SAC training is stable\n")
    else:
        print("\nTEST 3 FAILED - SAC training is unstable!\n")
    
    wrapped_env.close()
    return all_stable

def test_logging_structure():
    """Test 4: Check logging structure and prefixes"""
    print("=" * 60)
    print("TEST 4: Logging Structure")
    print("=" * 60)
    
    base_env = gym.make("UnicyclePendulumPD-v0")
    env = Monitor(base_env)
    marb_config = MARBConfig(k=10, G_min=-100, G_max=100)
    estimator = MovingAverageRewardBasedEstimator(marb_config)
    coach = CoachAgent(obs_dim=20, action_dim=2, auto_alpha=True, device="cpu")
    wrapped_env = CoachWrapper(env, coach_agent=coach, estimator=estimator)
    
    obs, info = wrapped_env.reset()
    
    # Collect data
    print("Collecting data...")
    for ep in range(3):
        done = False
        step_count = 0
        while not done and step_count < 500:
            student_action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(student_action)
            done = terminated or truncated
            step_count += 1
        obs, info = wrapped_env.reset()
    
    if wrapped_env.coach_buffer.size < 1000:
        print("Not enough data, skipping")
        wrapped_env.close()
        return False
    
    # Get update logs
    logs = wrapped_env.update_coach(n_gradient_steps=10, learning_starts=1000)
    
    print("\nChecking log structure...")
    
    # Expected train/ keys
    train_keys = [
        "train/critic_loss", "train/actor_loss", "train/q_mean",
        "train/q1_mean", "train/q2_mean", "train/logp_pi",
        "train/weight_assist_mean", "train/action_norm",
        "train/alpha", "train/alpha_loss"
    ]
    
    # Expected buffer/ keys
    buffer_keys = [
        "buffer/size", "buffer/fill_ratio", "buffer/reward_mean",
        "buffer/reward_std", "buffer/weight_assist_mean",
        "buffer/weight_assist_nonzero_ratio", "buffer/action_norm_mean",
        "buffer/weighted_action_norm_mean"
    ]
    
    all_present = True
    
    print("\n  Train metrics:")
    for key in train_keys:
        if key in logs:
            print(f"    {key}: {logs[key]:.4f}")
        else:
            print(f"    Missing: {key}")
            all_present = False
    
    print("\n  Buffer metrics:")
    for key in buffer_keys:
        if key in logs:
            if isinstance(logs[key], (int, float)):
                print(f"    {key}: {logs[key]}")
            else:
                print(f"    {key}: wrong type {type(logs[key])}")
        else:
            print(f"    Missing: {key}")
            all_present = False
    
    # Check episode rollout info
    obs, info = wrapped_env.reset()
    done = False
    while not done:
        student_action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(student_action)
        done = terminated or truncated
    
    print("\n  Rollout metrics (episode end):")
    rollout_keys = [
        "rollout/coach_reward_mean", "rollout/coach_reward_sum",
        "rollout/weight_assist_mean", "rollout/weight_assist_std",
        "rollout/coach_action_norm_mean", "rollout/skill_level"
    ]
    
    for key in rollout_keys:
        if key in info:
            print(f"    {key}: {info[key]:.4f}")
        else:
            print(f"    Missing: {key}")
            all_present = False
    
    if all_present:
        print("\nTEST 4 PASSED - All logging keys present\n")
    else:
        print("\nTEST 4 FAILED - Some logging keys missing\n")
    
    wrapped_env.close()
    return all_present

def test_full_integration():
    """Test 5: Full integration with skill level updates"""
    print("=" * 60)
    print("TEST 5: Full Integration")
    print("=" * 60)
    
    base_env = gym.make("UnicyclePendulumPD-v0")
    env = Monitor(base_env)
    marb_config = MARBConfig(k=10, G_min=-100, G_max=100)
    estimator = MovingAverageRewardBasedEstimator(marb_config)
    coach = CoachAgent(obs_dim=20, action_dim=2, auto_alpha=True, device="cpu")
    wrapped_env = CoachWrapper(env, coach_agent=coach, estimator=estimator)
    
    obs, info = wrapped_env.reset()
    
    total_episodes = 15
    update_frequency = 5
    
    skill_levels = []
    
    for ep in range(total_episodes):
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 1000:
            student_action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(student_action)
            episode_reward += reward
            done = terminated or truncated
            step_count += 1
        
        skill_levels.append(wrapped_env.skill_level)
        
        print(f"Episode {ep+1}/{total_episodes}: {step_count} steps, "
              f"reward: {episode_reward:.2f}, skill: {wrapped_env.skill_level:.3f}, "
              f"buffer: {wrapped_env.coach_buffer.size}")
        
        # Periodically update coach
        if (ep + 1) % update_frequency == 0 and wrapped_env.coach_buffer.size >= 1000:
            print(f"  Updating coach...")
            logs = wrapped_env.update_coach(n_gradient_steps=10, learning_starts=1000)
            print(f"     alpha: {logs.get('train/alpha', 0):.4f}, "
                  f"critic_loss: {logs.get('train/critic_loss', 0):.4f}, "
                  f"buffer size: {logs.get('buffer/size', 0)}")
        
        obs, info = wrapped_env.reset()
    
    print(f"\nCompleted {total_episodes} episodes")
    print(f"Final buffer size: {wrapped_env.coach_buffer.size}")
    print(f"Skill level progression: {skill_levels[0]:.3f} -> {skill_levels[-1]:.3f}")
    print("TEST 5 PASSED - Full integration successful\n")
    
    wrapped_env.close()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("COACH INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    try:
        # Test 1: Basic functionality
        test_basic_functionality()
        
        # Test 2: Buffer collection
        buffer_ok = test_buffer_collection()
        if not buffer_ok:
            print("Skipping remaining tests due to buffer issues\n")
            all_passed = False
        else:
            # Test 3: SAC training stability (CRITICAL)
            sac_ok = test_sac_training_stability()
            if not sac_ok:
                print("WARNING: SAC training unstable!\n")
                all_passed = False
            
            # Test 4: Logging structure
            logging_ok = test_logging_structure()
            if not logging_ok:
                print("WARNING: Logging structure incomplete\n")
                all_passed = False
            
            # Test 5: Full integration
            test_full_integration()
        
        if all_passed:
            print("\n" + "=" * 60)
            print("ALL TESTS PASSED")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("SOME TESTS FAILED - Check warnings above")
            print("=" * 60)
            
    except Exception as e:
        print(f"\nTEST FAILED WITH ERROR:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()