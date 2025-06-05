# experiment_runner.py (based on student_train.py)

import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import random
import dearpygui.dearpygui as dpg

from envs import unicycle
from hse.gamepad import Gamepad

##########################################################################################################################
##########################################################################################################################
######################## Human subject experiment runner code for the Unicycle Project ###################################
##########################################################################################################################
##########################################################################################################################

ENABLE_UI = True
# ENABLE_UI = False

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

# Timestamp for run identification
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"student_{timestamp}"

# Detailed structure for models and logs
logs_dir = os.path.join(data_dir, "logs", run_name)

# Create directories if not exist
[os.makedirs(d, exist_ok=True) for d in [data_dir, logs_dir]]


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
        motor_gain * -gamepad["axes"][1],
        roll_gain * gamepad["axes"][2],  # roll?
        roll_gain * -gamepad["axes"][3],  # pitch?
        yaw_gain * (gamepad["axes"][4] - gamepad["axes"][5]),  # yaw
    ]
    return action


# 4) Test function
def test(env):
    gamepad = Gamepad()
    if ENABLE_UI:
        # Initialize GUI
        dpg.create_context()
        dpg.create_viewport(
            title="Unicycle Experiment Simulation UI", width=800, height=600
        )

        # Create windows and plots
        with dpg.window(label="Actions", width=780, height=580):
            with dpg.plot(label="Action Values", height=550):
                dpg.add_plot_axis(dpg.mvXAxis, label="steps")
                dpg.add_plot_axis(dpg.mvYAxis, label="action values")
                dpg.add_line_series([], [], parent=dpg.last_item(), tag="action_plot")

    # Setup data storage
    actions = []
    x_data = []
    step_count = 0

    # Setup and show GUI
    dpg.setup_dearpygui()
    dpg.show_viewport()

    obs = env.reset()

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

            action = gamepad_to_action(gamepad_state)
            obs, reward, done, truncated, info = env.step(action)
            #print("Observation shape:", obs.shape)
            #print("Observation:", obs)

            # Store data for visualization
            step_count += 1
            x_data.append(step_count)
            actions.append(action)

            # Update action plot (show all 4 action dimensions)
            for i in range(4):
                action_values = [a[i] for a in actions[-100:]]  # Show recent actions
                action_steps = x_data[-len(action_values) :]
                dpg.set_value("action_plot", [action_steps, action_values])

            if done or truncated:
                obs = env.reset()
                done = False
                truncated = False

            if env.render_mode == "human":
                env.render()

            # Render GUI frame
            dpg.render_dearpygui_frame()

    except KeyboardInterrupt:
        print("Test interrupted (Ctrl + C).")
    finally:
        env.close()
        dpg.destroy_context()


# 5) Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform a human subject trial.")
    parser.add_argument("-e", "--env", action="store_true", help="Environment to use.")
    parser.add_argument(
        "--timesteps", type=int, default=10**10, help="Number of timesteps."
    )
    args = parser.parse_args()

    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    test_env = gym.make(
        "Student-v0",
        render_mode="human",
        frame_skip=FRAME_SKIP,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
    )
    test(test_env)
    test_env.close()
