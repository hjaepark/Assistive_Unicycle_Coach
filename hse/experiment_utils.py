# hse/experiment_utils.py

import csv
import numpy as np
from pathlib import Path


# CSV Field Definitions
TRAJECTORY_FIELDS = [
    "participant_id", "condition", "phase", "timestamp", "episode", "num_steps",
    *[f"obs_{i}" for i in range(19)],  # 19D observation
    "unicycle_roll", "unicycle_pitch", "pendulum_ang",
    "human_action_0", "human_action_1",
    "coach_action_0", "coach_action_1",
    "weight_assist",
    "effective_intervention_norm",
    "final_action_0", "final_action_1", "final_action_2", "final_action_3",
    "reward", "done", "truncated", "num_success", "success_indicator", "is_fallen",
    "skill_level"
]

SUMMARY_FIELDS = [
    "participant_id", "condition", "phase", "success_count", "cumulative_reward",
    "num_falls", "num_episodes", "time_to_3_successes", "avg_distance_traveled",
    "episode_duration", "balance_stability", "pendulum_stability"
]


def setup_data_storage(data_dir, participant_id, condition, phase):
    """
    Setup output directories and files
    
    Returns:
        tuple: (output_dir, trajectory_file, summary_file)
    """
    output_dir = Path(data_dir) / condition / participant_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trajectory_file = output_dir / f"{participant_id}_{phase}_trajectory.csv"
    summary_file = output_dir / f"{participant_id}_{phase}_summary.csv"
    
    return output_dir, trajectory_file, summary_file


def initialize_summary_data(participant_id, condition, phase, phase_timesteps):
    """Initialize summary data dictionary"""
    return {
        "participant_id": participant_id,
        "condition": condition,
        "phase": phase,
        "success_count": 0,
        "cumulative_reward": 0.0,
        "num_falls": 0,
        "num_episodes": 0,
        "time_to_3_successes": phase_timesteps / 200.0,
        "avg_distance_traveled": 0.0,
        "episode_duration": 0.0,
        "balance_stability": 0.0,
        "pendulum_stability": 0.0
    }


def create_trajectory_row(participant_id, condition, phase, timestamp, obs, 
                          human_action, info, reward, done, truncated, 
                          success_indicator, skill_level):
    """
    Create a single trajectory row for CSV
    
    Args:
        participant_id: Participant ID
        condition: Experiment condition
        phase: Experiment phase
        timestamp: Time since phase start
        obs: Environment observation (19D)
        human_action: Human's action (2D)
        info: Environment info dict
        reward: Reward
        done: Episode done flag
        truncated: Episode truncated flag
        success_indicator: Success on this step
        skill_level: Current skill level
    
    Returns:
        dict: Trajectory row
    """
    final_action = info.get("final_action", np.concatenate([human_action, [0.0, 0.0]]))

    return {
        "participant_id": participant_id,
        "condition": condition,
        "phase": phase,
        "timestamp": timestamp,
        "episode": info["num_episodes"],
        "num_steps": info["num_steps"],
        **{f"obs_{i}": obs[i] for i in range(19)},
        "unicycle_roll": info["unicycle_roll"],
        "unicycle_pitch": info["unicycle_pitch"],
        "pendulum_ang": info["pendulum_ang"],
        "human_action_0": human_action[0],
        "human_action_1": human_action[1],
        "coach_action_0": info.get("coach_action", np.zeros(2))[0],
        "coach_action_1": info.get("coach_action", np.zeros(2))[1],
        "weight_assist": info.get("weight_assist", 0.0),
        "effective_intervention_norm": info.get("effective_intervention_norm", 0.0),
        "final_action_0": final_action[0],
        "final_action_1": final_action[1],
        "final_action_2": final_action[2],
        "final_action_3": final_action[3],
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "num_success": info["num_success"],
        "success_indicator": success_indicator,
        "is_fallen": info["is_fallen"],
        "skill_level": skill_level
    }


def update_summary_statistics(summary_data, info, reward, current_max_x,
                              upright_rewards, pendulum_angles):
    """
    Update summary statistics
    
    Args:
        summary_data: Summary data dict (modified in-place)
        info: Environment info dict
        reward: Current reward
        current_max_x: Max x position in current episode
        upright_rewards: List of upright rewards
        pendulum_angles: List of pendulum angles
    """
    summary_data["cumulative_reward"] += reward
    
    # Upright reward: exp(-4.0 * (roll^2 + pitch^2 + pendulum_ang^2))
    upright_reward = np.exp(-4.0 * (
        info["unicycle_roll"]**2 + 
        info["unicycle_pitch"]**2 + 
        info["pendulum_ang"]**2
    ))
    upright_rewards.append(upright_reward)
    pendulum_angles.append(abs(info["pendulum_ang"]))


def finalize_summary_statistics(summary_data, episode_max_x, episode_steps,
                                upright_rewards, pendulum_angles):
    """
    Compute final summary statistics
    
    Args:
        summary_data: Summary data dict (modified in-place)
        episode_max_x: List of max x per episode
        episode_steps: List of steps per episode
        upright_rewards: List of all upright rewards
        pendulum_angles: List of all pendulum angles
    """
    summary_data["avg_distance_traveled"] = np.mean(episode_max_x) if episode_max_x else 0.0
    summary_data["episode_duration"] = np.mean(episode_steps) if episode_steps else 0.0
    summary_data["balance_stability"] = np.mean(upright_rewards) if upright_rewards else 0.0
    summary_data["pendulum_stability"] = np.mean(pendulum_angles) if pendulum_angles else 0.0


def save_summary_data(summary_file, summary_data):
    """Save summary data to CSV"""
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(summary_data)


def check_required_info_keys(info):
    """
    Check if environment info has all required keys
    
    Raises:
        KeyError: If required keys are missing
    """
    required_keys = [
        "num_steps", "num_episodes", "x_position", "unicycle_roll",
        "unicycle_pitch", "pendulum_ang", "num_success", "is_fallen"
    ]
    missing_keys = [key for key in required_keys if key not in info]
    if missing_keys:
        raise KeyError(f"Environment info missing keys: {missing_keys}")


def print_phase_header(phase, condition, participant_id):
    """Print phase start header"""
    print(f"\n{'='*70}")
    print(f"Phase: {phase} | Condition: {condition} | Participant: {participant_id}")
    print(f"{'='*70}\n")


def print_progress(total_steps, total_timesteps, skill_level, elapsed_time, 
                  success_count, num_falls):
    """Print progress update"""
    print(f"Step: {total_steps}/{total_timesteps} | "
          f"Skill: {skill_level:.3f} | "
          f"Elapsed: {elapsed_time:.1f}s | "
          f"Success: {success_count} | "
          f"Falls: {num_falls}")


def print_phase_complete(summary_data, trajectory_file, summary_file):
    """Print phase completion summary"""
    print(f"\n{'='*70}")
    print(f"Phase Complete!")
    print(f"Success: {summary_data['success_count']} | Falls: {summary_data['num_falls']}")
    print(f"Episodes: {summary_data['num_episodes']}")
    print(f"Trajectory: {trajectory_file}")
    print(f"Summary: {summary_file}")
    print(f"{'='*70}\n")


# === NEW: Action debugging functions ===

def print_action_header(coach_enabled):
    """Print header for action details"""
    if coach_enabled:
        print(f"{'Step':>6} | {'H_wheel':>7} {'H_pitch':>7} | "
              f"{'C_wheel':>7} {'C_pitch':>7} | {'Weight':>6} | {'Interv':>7} | {'Skill':>6}")
        print(f"{'-'*6}-+-{'-'*7}-{'-'*7}-+-{'-'*7}-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}")
    else:
        print(f"{'Step':>6} | {'H_wheel':>7} {'H_pitch':>7}")
        print(f"{'-'*6}-+-{'-'*7}-{'-'*7}")


def print_action_step(total_steps, human_action, coach_action, weight_assist,
                     effective_intervention_norm, skill_level, coach_enabled):
    """
    Print single step action details in compact format
    
    Args:
        total_steps: Current step count
        human_action: Human's action (2D: [wheel, pitch])
        coach_action: Coach's action (2D: [wheel, pitch])
        weight_assist: Coach's assistance weight (0-1)
        effective_intervention_norm: L2 norm of effective intervention
        skill_level: Current skill level estimate
        coach_enabled: Whether coach is active
    """
    if coach_enabled:
        print(f"{total_steps:6d} | "
              f"{human_action[0]:+7.3f} {human_action[1]:+7.3f} | "
              f"{coach_action[0]:+7.3f} {coach_action[1]:+7.3f} | "
              f"{weight_assist:6.3f} | "
              f"{effective_intervention_norm:7.4f} | "
              f"{skill_level:6.3f}")
    else:
        print(f"{total_steps:6d} | "
              f"{human_action[0]:+7.3f} {human_action[1]:+7.3f}")