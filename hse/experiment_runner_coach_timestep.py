# experiment_runner_coach_timestep.py (based on student_train.py)

import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import random
import time

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

pre_trained_dir = os.path.join(current_dir,"Assistive_Unicycle_Coach", "pre_trained_models")

# Timestamp for run identification
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"human_coach_timestep_{timestamp}"

# Detailed structure for models and logs
logs_dir = os.path.join(data_dir, "logs", run_name)

# Create directories if not exist
[os.makedirs(d, exist_ok=True) for d in [data_dir, logs_dir, pre_trained_dir]]

class TimestepAlphaUpdater:
    def __init__(self, alpha_init=0.0, alpha_final=0.0, total_timesteps=55920):
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


# 4) Test function
def test(env, coach_model_path, alpha_init, alpha_final, total_timesteps):
    gamepad = Gamepad()

    test_env = ActionMergeWrapper(
        env=env,
        coach_model_path=coach_model_path,
        alpha=alpha_init
    )
    
    alpha_updater = TimestepAlphaUpdater(alpha_init, alpha_final, total_timesteps)
    
    obs, _ = test_env.reset()

    #print("Observation shape:", obs.shape)
    #print("Observation:", obs)

    done = False
    truncated = False

    start_time = time.time() # Start timer
    last_log_step = 0
    last_success_count = 0

    try:
        while alpha_updater.current_step < total_timesteps:
            # action = env.gamepad_to_action(gamepad.get_state())
            gamepad_state = gamepad.get_state()
            if not gamepad_state:
                continue

            human_action = gamepad_to_action(gamepad_state)
            current_alpha = alpha_updater.update_alpha()
            alpha_updater.set_alpha(test_env)
            obs, reward, done, truncated, info = test_env.step(human_action)

            # To check step rate
            if alpha_updater.current_step % 1000 == 0:
                elapsed_time = time.time() - start_time
                steps_since_last = alpha_updater.current_step - last_log_step
                step_rate = steps_since_last / elapsed_time if elapsed_time > 0 else 0
                print(f"Step : {alpha_updater.current_step}, Alpha: {current_alpha:.3f}, "
                      f"Elapsed: {elapsed_time:.1f}s, Step Rate: {step_rate:.1f} steps/s")
                start_time = time.time() # Reset
                last_log_step = alpha_updater.current_step

            if alpha_updater.current_step % 100 == 0:
                #print(f"Step: {alpha_updater.current_step}, Alpha: {current_alpha:.3f}")
                print(f"Human Action: {human_action}")
                print(f"Coach Action: {info['coach_action']}")
                print(f"Final Action: {human_action + info['coach_action_prime']}")
                print(f"Reward: {reward}")
                print(f"X position: {info['x_position']:.3f}")
            #print("Observation shape:", obs.shape)
            #print("Observation:", obs)

            # Log success count changes
            current_success_count = info.get("num_success", 0)
            if current_success_count > last_success_count:
                print(f"Success! Total Success Count: {current_success_count} at Step: {alpha_updater.current_step}")
                last_success_count = current_success_count

            if done or truncated:
                obs, _ = test_env.reset()
                done = False
                truncated = False
                #alpha_updater.current_step = 0

            if test_env.render_mode == "human":
                test_env.render()
        print(f"Experiment completed: Reached {total_timesteps} timesteps. Final Alpha: {current_alpha:.3f}")
    except KeyboardInterrupt:
        print("Test interrupted (Ctrl + C).")
    finally:
        test_env.close()


# 5) Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform a human subject trial with coach assistance (timestep based method).")
    parser.add_argument("-e", "--env", action="store_true", help="Environment to use.")
    parser.add_argument(
        "--timesteps", type=int, default=55920, help="Total timesteps for alpha decay."
    )
    # 120000 for ideal 1x real time, 60000 for ideal 0.5x real time, 55920 for current 0.5x real time
    parser.add_argument("--coach_model_path", type=str, default=None, help="Path to pre-trained coach model.")
    parser.add_argument("--alpha_init", type=float, default=0.0, help="Initial alpha value.")
    parser.add_argument("--alpha_final", type=float, default=0.0, help="Final alpha value.")
    args = parser.parse_args()

    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # Default coach model path
    if args.coach_model_path is None:
        default_coach = os.path.join(pre_trained_dir, "unicycle_trained.zip")
        args.coach_model_path = default_coach

    base_env = gym.make(
        "Unicycle-v0",
        render_mode="human",
        frame_skip=FRAME_SKIP,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
    )
    test(base_env, args.coach_model_path, args.alpha_init, args.alpha_final, args.timesteps)
    base_env.close()
