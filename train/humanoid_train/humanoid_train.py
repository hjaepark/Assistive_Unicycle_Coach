# humanoid_train.py

import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import random
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


##########################################################################################################################
##########################################################################################################################
#################################### Humanoid-v5 agent environment training code  ########################################
##########################################################################################################################
##########################################################################################################################

##########################################################################################################################
# 1) Basic directory setup
##########################################################################################################################
current_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
data_dir = os.path.join(current_dir, "data")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"humanoid_{timestamp}"

models_dir = os.path.join(data_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model", run_name)
final_model_dir = os.path.join(models_dir, "final_model")
logs_dir = os.path.join(data_dir, "logs", run_name)

directories = [data_dir, models_dir, best_model_dir, final_model_dir, logs_dir]
for d in directories:
    os.makedirs(d, exist_ok=True)


##########################################################################################################################
# 2) Environment creation function
##########################################################################################################################
def make_humanoid_env(
    seed=0,
    forward_reward_weight=1.25,
    ctrl_cost_weight=0.1,
    contact_cost_weight=5e-7,
    contact_cost_range=(-np.inf, 10.0),
    healthy_reward=5.0,
    terminate_when_unhealthy=True,
    healthy_z_range=(1.0, 2.0),
    reset_noise_scale=1e-2,
    exclude_current_positions_from_observation=True,
    include_cinert_in_observation=True,
    include_cvel_in_observation=True,
    include_qfrc_actuator_in_observation=True,
    include_cfrc_ext_in_observation=True,
    render_mode=None
):
    """
    A closure that creates an environment compatible with both SubprocVecEnv and DummyVecEnv.
    """
    def _init():
        env = gym.make(
            "Humanoid-v5",
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=contact_cost_weight,
            contact_cost_range=contact_cost_range,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            include_cinert_in_observation=include_cinert_in_observation,
            include_cvel_in_observation=include_cvel_in_observation,
            include_qfrc_actuator_in_observation=include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
            render_mode=render_mode
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


##########################################################################################################################
# 3) Train function (with EvalCallback)
##########################################################################################################################
def train(
    total_timesteps,
    num_envs=8,
    device="cuda",
    forward_reward_weight=1.25,
    ctrl_cost_weight=0.1,
    contact_cost_weight=5e-7,
    contact_cost_range=(-np.inf, 10.0),
    healthy_reward=5.0,
    terminate_when_unhealthy=True,
    healthy_z_range=(1.0, 2.0),
    reset_noise_scale=1e-2,
    exclude_current_positions_from_observation=True,
    include_cinert_in_observation=True,
    include_cvel_in_observation=True,
    include_qfrc_actuator_in_observation=True,
    include_cfrc_ext_in_observation=True
):
    """
    Train the SAC model on a Humanoid-v5 environment using SubprocVecEnv for parallelization.
    Periodically evaluate using EvalCallback and save the best model to best_model_dir.
    """
    # 3.1 Create parallel training environment (SubprocVecEnv)
    train_env_fns = []
    for i in range(num_envs):
        train_env_fns.append(make_humanoid_env(
            seed=i,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=contact_cost_weight,
            contact_cost_range=contact_cost_range,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            include_cinert_in_observation=include_cinert_in_observation,
            include_cvel_in_observation=include_cvel_in_observation,
            include_qfrc_actuator_in_observation=include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation=include_cfrc_ext_in_observation
        ))
    vec_train_env = SubprocVecEnv(train_env_fns)

    # 3.2 Create evaluation environment (DummyVecEnv). Single environment by default, or you can stack multiple.
    eval_env_fns = [make_humanoid_env(
        seed=9999,  # Different seed for evaluation
        forward_reward_weight=forward_reward_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        contact_cost_weight=contact_cost_weight,
        contact_cost_range=contact_cost_range,
        healthy_reward=healthy_reward,
        terminate_when_unhealthy=terminate_when_unhealthy,
        healthy_z_range=healthy_z_range,
        reset_noise_scale=reset_noise_scale,
        exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        include_cinert_in_observation=include_cinert_in_observation,
        include_cvel_in_observation=include_cvel_in_observation,
        include_qfrc_actuator_in_observation=include_qfrc_actuator_in_observation,
        include_cfrc_ext_in_observation=include_cfrc_ext_in_observation
    )]
    vec_eval_env = DummyVecEnv(eval_env_fns)

    # 3.3 Create SAC model
    model = SAC(
        'MlpPolicy',
        vec_train_env,
        verbose=1,
        device=device,
        batch_size=256,
        buffer_size=1_000_000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        tensorboard_log=logs_dir
    )

    # 3.4 Configure EvalCallback
    eval_callback = EvalCallback(
        vec_eval_env,
        best_model_save_path=best_model_dir,
        log_path=logs_dir,
        eval_freq=10_000,          # Evaluate every 10,000 steps
        n_eval_episodes=5,         # Average performance over 5 episodes
        deterministic=True,
        render=False
    )

    # 3.5 Train model
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="SAC_Humanoid",
        callback=eval_callback
    )

    # 3.6 Save the final model
    final_model_path = os.path.join(final_model_dir, f"humanoid_{timestamp}_final")
    model.save(final_model_path)
    print(f"[TRAIN] Training finished. Final model saved at: {final_model_path}")

    vec_train_env.close()
    vec_eval_env.close()


##########################################################################################################################
# 4) Test function
##########################################################################################################################
def test(path_to_model,
         render_mode='human',
         forward_reward_weight=1.25,
         ctrl_cost_weight=0.1,
         contact_cost_weight=5e-7,
         contact_cost_range=(-np.inf, 10.0),
         healthy_reward=5.0,
         terminate_when_unhealthy=True,
         healthy_z_range=(1.0, 2.0),
         reset_noise_scale=1e-2,
         exclude_current_positions_from_observation=True,
         include_cinert_in_observation=True,
         include_cvel_in_observation=True,
         include_qfrc_actuator_in_observation=True,
         include_cfrc_ext_in_observation=True):
    """
    Load the trained SAC model (.zip) and run it on a single Humanoid-v5 environment in an infinite loop.
    Press Ctrl + C to exit.
    """
    model = SAC.load(path_to_model)
    print(f"[TEST] Loaded model from: {path_to_model}")

    # Create a single test environment
    test_env = gym.make(
        "Humanoid-v5",
        render_mode=render_mode,
        forward_reward_weight=forward_reward_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        contact_cost_weight=contact_cost_weight,
        contact_cost_range=contact_cost_range,
        healthy_reward=healthy_reward,
        terminate_when_unhealthy=terminate_when_unhealthy,
        healthy_z_range=healthy_z_range,
        reset_noise_scale=reset_noise_scale,
        exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        include_cinert_in_observation=include_cinert_in_observation,
        include_cvel_in_observation=include_cvel_in_observation,
        include_qfrc_actuator_in_observation=include_qfrc_actuator_in_observation,
        include_cfrc_ext_in_observation=include_cfrc_ext_in_observation
    )

    obs, info = test_env.reset()
    done = False
    truncated = False

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _info = test_env.step(action)

            if done or truncated:
                obs, info = test_env.reset()
                done = False
                truncated = False

            if render_mode == 'human':
                test_env.render()

    except KeyboardInterrupt:
        print("[TEST] Test interrupted (Ctrl + C).")
    finally:
        test_env.close()


##########################################################################################################################
# 5) Main
##########################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the Humanoid SAC model with EvalCallback.')
    parser.add_argument('-train', '--train', action='store_true', help='Train the model.')
    parser.add_argument('-test', '--test', type=str, help='Test the model (path to .zip file).')
    parser.add_argument('--timesteps', type=int, default=2_000_000, help='Number of timesteps for training.')
    parser.add_argument('--num_envs', type=int, default=8, help='Number of parallel environments for training.')
    args = parser.parse_args()

    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if args.train:
        train(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            device="cuda",  # Use GPU
            forward_reward_weight=1.25,
            ctrl_cost_weight=0.1,
            contact_cost_weight=5e-7,
            contact_cost_range=(-np.inf, 10.0),
            healthy_reward=5.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(1.0, 2.0),
            reset_noise_scale=1e-2,
            exclude_current_positions_from_observation=True,
            include_cinert_in_observation=True,
            include_cvel_in_observation=True,
            include_qfrc_actuator_in_observation=True,
            include_cfrc_ext_in_observation=True
        )

    if args.test:
        test(
            path_to_model=args.test,
            render_mode='human',
            forward_reward_weight=1.25,
            ctrl_cost_weight=0.1,
            contact_cost_weight=5e-7,
            contact_cost_range=(-np.inf, 10.0),
            healthy_reward=5.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(1.0, 2.0),
            reset_noise_scale=1e-2,
            exclude_current_positions_from_observation=True,
            include_cinert_in_observation=True,
            include_cvel_in_observation=True,
            include_qfrc_actuator_in_observation=True,
            include_cfrc_ext_in_observation=True
        )
