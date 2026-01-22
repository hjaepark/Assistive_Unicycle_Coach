# hse/experiment.py

import argparse
from cmath import phase
import gymnasium as gym
import numpy as np
import random
import time
import csv
from pathlib import Path
import torch

import mujoco

from envs import unicycle
from hse.gamepad import Gamepad
from hse import experiment_utils as utils

from train.unicycle_train.coach.coach_agent import CoachAgent
from train.unicycle_train.coach.coach_config import CoachRewardConfig
from train.unicycle_train.cotraining.config_loader import load_config
from train.unicycle_train import path as P

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "lookat": np.array((2.5, 0.0, 0.0)),
    "distance": 5,
    "azimuth": 45.0,
    "elevation": -15.0,
}
FRAME_SKIP = 1

PHASE_TIMESTEPS = {
    "pre-test": 13980,
    "training": 55920,
    "post-test": 13980,
}

HSE_DATA_DIR = P.PROJECT_ROOT / "data"
HSE_DATA_DIR.mkdir(parents=True, exist_ok=True)

class HSECoachWrapper(gym.Wrapper):
    
    def __init__(self, env, coach_model_path=None, estimator=None,
                 reward_config: CoachRewardConfig = None):
        super().__init__(env)
        
        self.coach_agent = None
        self.estimator = estimator
        self.skill_level = 0.0
        
        self.reward_cfg = reward_config or CoachRewardConfig()
        self.kappa = self.reward_cfg.kappa
        
        if coach_model_path:
            self.coach_agent = self._load_coach(coach_model_path)
            print(f"[info] Coach loaded: {coach_model_path}")
            print(f"[info] Initial skill: {self.skill_level:.2f}")
        
        self.episode_rewards = []
        self.episode_return = 0.0

        self._last_obs = None
        
        self.rollout_weight_assists = []
        self.rollout_effective_intervention_norms = []
    
    # TODO Might need to make this to be able to load from the config file to prevent hard coding
    def _load_coach(self, checkpoint_path):
        coach = CoachAgent(
            obs_dim=22,
            action_dim=2,
            lr=0.0003,  
            auto_alpha=True,
            device="cpu"
        )
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        coach.actor.load_state_dict(checkpoint['actor'])
        coach.critic1.load_state_dict(checkpoint['critic1'])
        coach.critic2.load_state_dict(checkpoint['critic2'])
        coach.actor.eval()
        
        return coach
    
    def step(self, human_action):
        info = {}
        
        if self.coach_agent:
            # Construct coach observation: [state(19), skill(1), human_action(2)]
            coach_obs = np.concatenate([
                self._last_obs,
                [self.skill_level],
                human_action
            ])
            
            # Get coach action and weight_assist
            coach_action, weight_assist = self.coach_agent.predict(coach_obs, deterministic=False)
            weight_assist = np.clip(weight_assist, 0.0, 1.0)
            
            # Compute interventions
            intended_intervention = weight_assist * self.kappa * coach_action
            exec_action = np.clip(human_action + intended_intervention, -1.0, 1.0)
            effective_intervention = exec_action - human_action
            
            # Store in info
            info["coach_action"] = coach_action
            info["weight_assist"] = weight_assist
            info["effective_intervention_norm"] = float(np.linalg.norm(effective_intervention))
            
            # Track statistics
            self.rollout_weight_assists.append(weight_assist)
            self.rollout_effective_intervention_norms.append(
                float(np.linalg.norm(effective_intervention))
            )
        else:
            # No coach
            exec_action = human_action
            info["coach_action"] = np.zeros(2, dtype=np.float32)
            info["weight_assist"] = 0.0
            info["effective_intervention_norm"] = 0.0
        
        # Execute in environment
        obs, reward, done, truncated, env_info = self.env.step(exec_action)
        info.update(env_info)
        
        # Track episode return for skill estimation
        self.episode_return += reward
        
        # Update skill level at episode end
        if done or truncated:
            self.episode_rewards.append(self.episode_return)

            if self.estimator and "episode" in info and "r" in info["episode"]:
                self.skill_level = self.estimator.update(info["episode"]["r"])
            
            # Add rollout statistics to info
            if self.rollout_weight_assists:
                info["rollout/weight_assist_mean"] = float(np.mean(self.rollout_weight_assists))
                info["rollout/weight_assist_std"] = float(np.std(self.rollout_weight_assists))
                info["rollout/effective_intervention_norm_mean"] = float(
                    np.mean(self.rollout_effective_intervention_norms)
                )
                info["rollout/skill_level"] = float(self.skill_level)
            
            # Reset for next episode
            self.episode_return = 0.0
            self.rollout_weight_assists = []
            self.rollout_effective_intervention_norms = []
        
        self._last_obs = obs
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        try:
            obs, info = self.env.reset(**kwargs)
        except ValueError:
            obs = self.env.reset(**kwargs)
            info = {"num_steps": 0, "num_episodes": 0}
        
        self.episode_return = 0.0
        self._last_obs = obs
        return obs, info

def gamepad_to_action(gamepad) -> np.ndarray:
    # Convert gamepad to action in unicycle environment
    motor_gain = 1.0
    pitch_gain = 1.0
    action = [
        motor_gain * -gamepad["axes"][1], # Wheel
        pitch_gain * gamepad["axes"][3], # Pitch
    ]
    return np.array(action, dtype=np.float32)

def run_phase(env, coach_model_path, participant_id, condition, phase, total_timesteps,
              estimator_config: dict, verbose_actions=False):
    gamepad = Gamepad()
    start_time = time.time()
    
    # Setup coach wrapper
    use_coach = (condition == "with-coach" and phase == "training")
    
    if use_coach:
        if coach_model_path is None or not Path(coach_model_path).exists():
            raise FileNotFoundError(f"Coach model not found: {coach_model_path}")
        
        # TODO Might need to make this to be able to load from the config file to prevent hard coding
        # Load reward config from optimized.yaml
        reward_config = CoachRewardConfig(
            rho=0.742,
            beta=0.0115,
            lambda_E=0.056,
            kappa=np.array([0.241, 0.241], dtype=np.float32)
        )

        if estimator_config['type'] == "MARB":
            from train.unicycle_train.coach.estimators.marb_estimator import (
                MARBConfig, MovingAverageRewardBasedEstimator, precompute_params
            )
            marb_cfg, info = precompute_params(P.DATA_STUDENT_DIR, k_candidates=range(5, 11))
            estimator = MovingAverageRewardBasedEstimator(marb_cfg)
            print(f"[info] MARB: k={marb_cfg.k}, G_min={info['G_min']:.2f}, G_max={info['G_max']:.2f}")
        elif estimator_config['type'] == "manual":
            class ManualEstimator:
                def __init__(self, skill):
                    self.skill = skill
                def update(self, episode_return):
                    return self.skill
            estimator = ManualEstimator(estimator_config['manual_skill_level'])
            print(f"[info] Manual estimator: skill={estimator_config['manual_skill_level']}")
        else:
            raise ValueError(f"Unknown estimator type: {estimator_config['type']}")
        
        wrapped_env = HSECoachWrapper(
            env=env,
            coach_model_path=coach_model_path,
            estimator=estimator,
            reward_config=reward_config
        )
    else:
        wrapped_env = HSECoachWrapper(env=env, coach_model_path=None)
    
    # Setup data storage using utils
    output_dir, trajectory_file, summary_file = utils.setup_data_storage(
        HSE_DATA_DIR, participant_id, condition, phase
    )
    
    # Initialize summary data using utils
    summary_data = utils.initialize_summary_data(
        participant_id, condition, phase, PHASE_TIMESTEPS[phase]
    )
    
    # Initialize tracking variables
    obs, info = wrapped_env.reset()
    if "num_steps" not in info:
        info["num_steps"] = 0
    if "num_episodes" not in info:
        info["num_episodes"] = 0
    
    trajectory_data = []
    episode_max_x = []
    episode_steps = []
    upright_rewards = []
    pendulum_angles = []
    last_success_count = 0
    current_max_x = 0.0
    current_episode_steps = 0
    time_to_3_successes = None
    total_steps = 0
    next_step_time = time.time()
    
    try:
        with open(trajectory_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=utils.TRAJECTORY_FIELDS)
            writer.writeheader()
            
            utils.print_phase_header(phase, condition, participant_id)
            
            # Print coach info
            if use_coach:
                print(f"[Coach] ENABLED - kappa: [{wrapped_env.reward_cfg.kappa[0]:.3f}, {wrapped_env.reward_cfg.kappa[1]:.3f}]")
            else:
                print(f"[Coach] DISABLED\n")
            
            # Print column headers for verbose output
            if verbose_actions:
                utils.print_action_header(use_coach)
            
            while total_steps < total_timesteps:
                step_start_time = time.time()
                
                # Get gamepad input
                gamepad_state = gamepad.get_state()
                if not gamepad_state:
                    print(f"[warning] Gamepad state is None at t={time.time() - start_time:.1f}s")
                    continue
                
                human_action = gamepad_to_action(gamepad_state)
                
                # Step environment
                obs, reward, done, truncated, info = wrapped_env.step(human_action)

                # UI for simulation
                try:
                    viewer = env.unwrapped.mujoco_renderer.viewer

                    if viewer:
                        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, "Condition:", str(condition))
                        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, "Phase:", str(phase))

                        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Success:", str(summary_data["success_count"]))
                        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Falls:", str(summary_data["num_falls"]))

                        elapsed_sec = time.time() - start_time
                        mins = int(elapsed_sec // 60)
                        secs = int(elapsed_sec % 60)
                        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, "Time:", f"{mins:02d}:{secs:02d}")
                    
                        current_x = info.get("x_position", 0.0)
                        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, "Distance:", f"{current_x:.2f} m")

                        if use_coach:
                            assist = info.get("weight_assist", 0.0)
                            viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, "Coach Help:", f"{assist*100:.0f}%")

                except AttributeError:
                    pass

                current_episode_steps += 1
                total_steps += 1
                
                # Check required info keys using utils
                utils.check_required_info_keys(info)
                
                if verbose_actions:
                    utils.print_action_step(
                        total_steps,
                        human_action,
                        info.get("coach_action", np.zeros(2)),
                        info.get("weight_assist", 0.0),
                        info.get("effective_intervention_norm", 0.0),
                        wrapped_env.skill_level,
                        use_coach
                    )
                
                # Create trajectory row using utils
                success_indicator = info["num_success"] > last_success_count
                trajectory_row = utils.create_trajectory_row(
                    participant_id, condition, phase,
                    time.time() - start_time,
                    obs, human_action, info,
                    reward, done, truncated,
                    success_indicator,
                    wrapped_env.skill_level
                )
                
                trajectory_data.append(trajectory_row)
                writer.writerow(trajectory_row)
                
                # Update summary statistics using utils
                current_max_x = max(current_max_x, info["x_position"])
                utils.update_summary_statistics(
                    summary_data, info, reward, current_max_x,
                    upright_rewards, pendulum_angles
                )
                
                # Success handling
                if success_indicator:
                    summary_data["success_count"] = info["num_success"]
                    last_success_count = info["num_success"]
                    if summary_data["success_count"] == 3 and time_to_3_successes is None:
                        time_to_3_successes = time.time() - start_time
                        summary_data["time_to_3_successes"] = time_to_3_successes
                    print(f"\n✓ Success #{summary_data['success_count']} at step {info['num_steps']}\n")
                
                # Fall handling
                if info["is_fallen"] and len(trajectory_data) > 1 and not trajectory_data[-2]["is_fallen"]:
                    summary_data["num_falls"] += 1
                
                # Episode end handling
                if done or truncated:
                    episode_max_x.append(current_max_x)
                    episode_steps.append(current_episode_steps)
                    summary_data["num_episodes"] += 1
                    
                    # Print episode summary if verbose
                    if verbose_actions and use_coach:
                        if wrapped_env.rollout_weight_assists:
                            print(f"\n[Episode {summary_data['num_episodes']}] "
                                  f"Avg assist: {np.mean(wrapped_env.rollout_weight_assists):.3f}, "
                                  f"Avg intervention: {np.mean(wrapped_env.rollout_effective_intervention_norms):.4f}\n")
                    
                    obs, info = wrapped_env.reset()
                    if "num_episodes" not in info:
                        info["num_episodes"] = summary_data["num_episodes"]
                    current_max_x = 0.0
                    current_episode_steps = 0
                
                # Print progress using utils (less frequently when verbose_actions is on)
                print_interval = 5000 if verbose_actions else 1000
                if total_steps % print_interval == 0:
                    print()  # Empty line for readability
                    utils.print_progress(
                        total_steps, total_timesteps,
                        wrapped_env.skill_level,
                        time.time() - start_time,
                        summary_data["success_count"],
                        summary_data["num_falls"]
                    )
                    print()  # Empty line for readability
                
                next_step_time += 0.005
                sleep_time = next_step_time - time.time()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -0.01:
                    if total_steps % 100 == 0:
                        print(f"[warning] Step {total_steps} behind schedule by {-sleep_time*1000:.1f} ms")
                    next_step_time = time.time()

        
        # Finalize summary using utils
        utils.finalize_summary_statistics(
            summary_data, episode_max_x, episode_steps,
            upright_rewards, pendulum_angles
        )
        
        # Save summary using utils
        utils.save_summary_data(summary_file, summary_data)
        
        # Print completion using utils
        utils.print_phase_complete(summary_data, trajectory_file, summary_file)
    
    except KeyboardInterrupt:
        print("\n[warning] Interrupted (Ctrl+C)")
        utils.save_summary_data(summary_file, summary_data)
        print(f"Partial summary saved: {summary_file}")
    
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSE with PyTorch Coach")
    parser.add_argument("--participant_id", type=str, required=True, help="Participant ID (e.g., P01)")
    parser.add_argument("--condition", type=str, choices=["with-coach", "without-coach"], required=True)
    parser.add_argument("--phase", type=str, choices=["pre-test", "training", "post-test"], required=True)
    parser.add_argument("--coach_model_path", type=str, default=None, help="Path to coach_agent.pth")
    parser.add_argument("--estimator_config", type=str, 
                    default="train/unicycle_train/cotraining/configs/optimized.yaml",
                    help="Path to config YAML (uses estimator section)")
    parser.add_argument("--debug", action="store_true",
                        help="Print action details every step for debugging")
    
    args = parser.parse_args()
    
    # Default coach model path using path.py
    if args.coach_model_path is None:
        try:
            latest_run_dir = max(
                P.LOGS_COTRAINING_DIR.glob("run_*"),
                key=lambda x: x.stat().st_mtime
            )
            args.coach_model_path = str(latest_run_dir / "models" / "coach_agent.pth")
            print(f"[info] Using latest coach: {args.coach_model_path}")
        except ValueError:
            if args.condition == "with-coach":
                print(f"[error] No coach models found in: {P.LOGS_COTRAINING_DIR}")
                exit(1)
    
    # Verify coach model
    if args.condition == "with-coach" and args.phase == "training":
        if not Path(args.coach_model_path).exists():
            print(f"[error] Coach model not found: {args.coach_model_path}")
            exit(1)
    
    # Load estimator config
    estimator_cfg = load_config(args.estimator_config)['estimator']
    print(f"[info] Loaded estimator config: {args.estimator_config}")
    
    # Set seeds
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment
    env = gym.make(
        "UnicyclePendulumPD-v0",
        render_mode="human",
        frame_skip=FRAME_SKIP,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
    )
    
    # Run phase
    run_phase(
        env,
        args.coach_model_path,
        args.participant_id,
        args.condition,
        args.phase,
        PHASE_TIMESTEPS[args.phase],
        estimator_config=estimator_cfg,
        verbose_actions=args.debug
    )