# experiment_runner.py (based on student_train.py)

import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import random

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


# Directory setup
current_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
data_dir = os.path.join(current_dir, "data")

pre_trained_dir = os.path.join(current_dir, "pre_trained_models")

# Timestamp for run identification
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"human_coach_{timestamp}"

# Detailed structure for models and logs
logs_dir = os.path.join(data_dir, "logs", run_name)

# Create directories if not exist
[os.makedirs(d, exist_ok=True) for d in [data_dir, logs_dir, pre_trained_dir]]


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


# 4) Test function
def test(env, coach_model_path, alpha):
    gamepad = Gamepad()

    test_env = ActionMergeWrapper(
        env=env,
        coach_model_path=coach_model_path,
        alpha=alpha
    )

    obs, _ = test_env.reset()

    #print("Observation shape:", obs.shape)
    #print("Observation:", obs)

    done = False
    truncated = False

    try:
        while True:
            # action = env.gamepad_to_action(gamepad.get_state())
            gamepad_state = gamepad.get_state()
            if not gamepad_state:
                continue

            human_action = gamepad_to_action(gamepad_state)
            obs, reward, done, truncated, info = test_env.step(human_action)

            print(f"Human Action: {human_action}")
            print(f"Coach Action: {info['coach_action']}")
            print(f"Final Action: {human_action + info['coach_action_prime']}")
            print(f"Reward: {reward}")

            if done or truncated:
                obs, _ = test_env.reset()
                done = False
                truncated = False

            if test_env.render_mode == "human":
                test_env.render()

    except KeyboardInterrupt:
        print("Test interrupted (Ctrl + C).")
    finally:
        env.close()


# 5) Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform a human subject trial with coach assistance.")
    parser.add_argument("-e", "--env", action="store_true", help="Environment to use.")
    parser.add_argument("--timesteps", type=int, default=10**10, help="Number of timesteps.")
    parser.add_argument("--coach_model_path", type=str, default=None, help="Path to pre-trained coach model.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fixed alpha for action merging.")
    args = parser.parse_args()

    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    if args.coach_model_path is None:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
        default_coach = os.path.join(project_root, "pre_trained_models", "unicycle_trained.zip")
        args.coach_model_path = default_coach

    base_env = gym.make(
        "Unicycle-v0",
        render_mode="human",
        frame_skip=FRAME_SKIP,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
    )
    test(base_env, args.coach_model_path, args.alpha)
    base_env.close()
