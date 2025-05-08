# experiment_runner_coach_timestep.py (based on student_train.py)

import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import random
import time
import csv
from pathlib import Path

from envs import unicycle
from hse.gamepad import Gamepad
from train.unicycle_train.wrapper.action_merging_wrapper import ActionMergeWrapper
from stable_baselines3 import SAC

##########################################################################################################################
##########################################################################################################################
######################## Human subject experiment runner code for the Unicycle Project ###################################
##########################################################################################################################
##########################################################################################################################

# Experiment parameters
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "azimuth": 45.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
}
FRAME_SKIP = 1

PHASE_TIMESTEPS = {
    "pre-test": 13980,
    "training": 55920,
    "post-test": 13980,
}

# Directory setup
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(current_dir, "Assistive_Unicycle_Coach", "data")

pre_trained_dir = os.path.join(current_dir, "Assistive_Unicycle_Coach", "pre_trained_models")

# # Timestamp for run identification
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# run_name = f"human_coach_timestep_{timestamp}"

# # Detailed structure for models and logs
# logs_dir = os.path.join(data_dir, "logs", run_name)

# # Create directories if not exist
# [os.makedirs(d, exist_ok=True) for d in [data_dir, logs_dir, pre_trained_dir]]

class TimestepAlphaUpdater:
    def __init__(self, alpha_init=0.7, alpha_final=0.0, total_timesteps=55920):
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.total_timesteps = total_timesteps
        self.current_step = 0
        self.current_alpha = alpha_init

    def update_alpha(self):
        self.current_step += 1
        ratio = min(self.current_step / float(self.total_timesteps), 1.0)
        self.current_alpha = self.alpha_init + ratio * (self.alpha_final - self.alpha_init)
        return self.current_alpha

    def set_alpha(self, env):
        if hasattr(env, "set_alpha"):
            env.set_alpha(self.current_alpha)

def gamepad_to_action(gamepad) -> np.ndarray:
    # Gains
    # zeros
    # motor_gain = 0.0
    # roll_gain = 0.0
    # yaw_gain = 0.0

    # ones
    motor_gain = 1.0
    # roll_gain = 1.0
    # yaw_gain = 1.0

    # hand-tuned
    # motor_gain = 0.50
    roll_gain = 1.0
    yaw_gain = 0.125

    action = [
        motor_gain * -gamepad["axes"][1], # Motor
        roll_gain * gamepad["axes"][2],  # Roll
        roll_gain * -gamepad["axes"][3],  # Pitch
        yaw_gain * (gamepad["axes"][4] - gamepad["axes"][5]),  # Yaw
    ]
    return np.array(action, dtype=np.float32)

def run_phase(env, coach_model_path, participant_id, condition, phase, total_timesteps):
    gamepad = Gamepad()

    if condition == "with-coach" and phase == "training" and not os.path.exists(coach_model_path):
        raise FileNotFoundError(f"Coach model not found: {coach_model_path}")
    
    # Set alpha based on condition
    alpha_init = 0.7 if condition == "with-coach" and phase == "training" else 0.0
    alpha_final = 0.0

    if condition == "with-coach":
        wrapped_env = ActionMergeWrapper(
            env=env,
            coach_model_path=coach_model_path if condition == "with-coach" else None,
            alpha=alpha_init
        )
    else:
        wrapped_env = env

    obs, _ = wrapped_env.reset()

    alpha_updater = TimestepAlphaUpdater(alpha_init, alpha_final, total_timesteps)

    # Data collection for preliminary evaluation
    summary_data = {
        "participant_id": participant_id,
        "condition": condition,
        "phase": phase,
        "success_count": 0,
        "cumulative_reward": 0.0,
        "num_failures": 0,
        "avg_distance_traveled": 0.0,
        "episode_count": 0
    }
    episode_rewards =[]
    episode_max_x = []

    done = False
    truncated = False
    start_time = time.time()
    last_success_count = 0
    current_max_x = 0.0

    try:
        while alpha_updater.current_step < total_timesteps:
            step_start_time = time.time()
            gamepad_state = gamepad.get_state()
            if not gamepad_state:
                continue

            human_action = gamepad_to_action(gamepad_state)
            current_alpha = alpha_updater.update_alpha()
            alpha_updater.set_alpha(wrapped_env)
            obs, reward, done, truncated, info = wrapped_env.step(human_action)

            summary_data["cumulative_reward"] += reward
            current_max_x = max(current_max_x, info["x_position"])
            if info["num_success"] > last_success_count:
                summary_data["success_count"] = info["num_success"]
                last_success_count = info["num_success"]
                print(f"Success! Total Success Count: {summary_data['success_count']} at Step: {alpha_updater.current_step}")

            if done or truncated:
                if not info["goal_reached"] and done:
                    summary_data["num_failures"] += 1
                summary_data["episode_count"] += 1
                episode_rewards.append(reward)
                episode_max_x.append(current_max_x)
                obs, _ = wrapped_env.reset()
                done = False
                truncated = False
                current_max_x = 0.0

            
            if alpha_updater.current_step % 100 == 0:
                print(f"Step: {alpha_updater.current_step}, Alpha: {current_alpha:.3f}")
                print(f"Human Action: {human_action}")
                print(f"Coach Action: {info.get('coach_action', np.zeros_like(human_action))}")
                print(f"Final Action: {human_action + info.get('coach_action_prime', np.zeros_like(human_action))}")
                print(f"Reward: {reward}")
                print(f"X position: {info['x_position']:.3f}")

            # Progress monitoring
            if alpha_updater.current_step % 1000 == 0:
                elapsed_time = time.time() - start_time
                step_rate = 1000 / elapsed_time if elapsed_time > 0 else 0
                print(f"Step: {alpha_updater.current_step}/{total_timesteps}, "
                      f"Apha: {current_alpha:.3f}, Elapsed: {elapsed_time:.1f}s, "
                      f"Step Rate: {step_rate:.1f} steps/s, "
                      f"Successes: {summary_data['success_count']}, "
                      f"Failures: {summary_data['num_failures']}"
                      )
                start_time = time.time()  # Reset timer
                if condition == "with-coach" and phase == "training":
                    print(f"Alpha decayed to: {current_alpha:.3f}")

            elapsed = time.time() - step_start_time
            sleep_time = max(0, 0.020 - elapsed)
            time.sleep(sleep_time)

        summary_data["avg_distance_traveled"] = np.mean(episode_max_x) if episode_max_x else 0.0

        # Saving data
        output_dir = Path(data_dir) / condition / participant_id
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / f"{phase}_summary.csv"
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data.keys())
            writer.writeheader()
            writer.writerow(summary_data)
        #print(f"Saved summary to {summary_file}")

    except KeyboardInterrupt:
        print("Experiment interrupted (Ctrl + C)")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data.keys())
            writer.writeheader()
            writer.writerow(summary_data)
        print(f"Saved partial summary to {summary_file}")
    finally:
        env.close()

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preliminary human subject experiment")
    parser.add_argument("--participant_id", type=str, required=True, help="Participant ID (P01, P02, ..)")
    parser.add_argument("--condition", type=str, choices=["with-coach", "without-coach"], required=True, help="Condition")
    parser.add_argument("--phase", type=str, choices=["pre-test", "training", "post-test"], required=True, help="Phase")
    parser.add_argument("--coach_model_path", type=str, default=None, help="Path to pre-trained coach model")
    args = parser.parse_args()

    # Default coach model path
    if args.coach_model_path is None:
        default_coach = os.path.join(pre_trained_dir, "unicycle_trained.zip")
        args.coach_model_path = default_coach

    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    base_env = gym.make(
        "Unicycle-v0",
        render_mode="human",
        frame_skip=FRAME_SKIP,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
    )

    total_timesteps = PHASE_TIMESTEPS[args.phase]
    run_phase(
        base_env,
        args.coach_model_path,
        args.participant_id,
        args.condition,
        args.phase,
        total_timesteps
    )
