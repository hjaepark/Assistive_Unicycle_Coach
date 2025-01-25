import os
import datetime
import time
import argparse
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import envs
from envs.student_v0 import Student
from envs.coach_student_wrapper import CoachStudentWrapper

# Using monitoring class
from train.monitoring.coach_student_monitor import CoachStudentMonitor
from train.monitoring.coach_monitor import CoachMonitorCallback
from train.monitoring.student_monitor import StudentMonitorCallback


##########################################################################################################################
##########################################################################################################################
######################## Student agent with Coach agent training code for the Unicycle Project ###########################
##########################################################################################################################
##########################################################################################################################

"""
Now using coach_student_wrapper
"""

# TEST

# Basic directory structure setup
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(current_dir, "data")

# Create timestamp directory for current run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"coach_student_{timestamp}"

# Setup detailed directory structure for current run
models_dir = os.path.join(data_dir, "models")
best_model_dir = os.path.join(models_dir, "best_model")
checkpoint_dir = os.path.join(models_dir, "checkpoints")
logs_dir = os.path.join(data_dir, "logs")

# Create all directories
directories = [
    data_dir, models_dir, best_model_dir, checkpoint_dir, logs_dir
]
for d in directories:
    os.makedirs(d, exist_ok=True)

# Maximum number of steps per episode
MAX_EPISODE_STEPS = 10000

# Success thresholds for model evaluation and saving
SUCCESS_THRESHOLDS = [1000]

def train_coach_student(env, coach_model_path, strategy: int = 1):
    
    """

    Main training function
    strategy:
      1 = Continuous
      2 = Performance-Based
      3 = Progressive Disengagement
      4 = Need-Based

    """

    """
    TODO
    Check both residual and partial residual policy to prove that they are equal in computing
    """

    # Apply coach+student wrapper
    wrapped_env = CoachStudentWrapper(
        env=env,
        coach_model_path=coach_model_path,
        mode="residual",
        alpha=0.5,
        reward_coef=0.01,
        reward_shaping=False
    )

    # Create evaluation environment
    eval_env = gym.make('Student-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    # Apply wrapper to evaluation environment
    wrapped_eval_env = CoachStudentWrapper(
        env=eval_env,
        coach_model_path=coach_model_path,
        mode="residual",
        alpha=0.5,
        reward_coef=0.01,
        reward_shaping=False
    )

    eval_callback = EvalCallback(
        wrapped_eval_env, 
        best_model_save_path=best_model_dir,
        log_path=logs_dir, 
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Using custom logger callback from student agent training code
    training_monitor = CoachStudentMonitor(data_dir)
    coach_monitor = CoachMonitorCallback(verbose=1)
    student_monitor = StudentMonitorCallback(verbose=1)

    # Combine callbacks
    callbacks = [eval_callback, training_monitor, coach_monitor, student_monitor]

    # Initialize the SAC policy
    model = SAC("MlpPolicy", wrapped_env, verbose=1, device="cuda", tensorboard_log=logs_dir)

    start_time = time.time()
    threshold_index = 0
    total_timesteps = 0

    while threshold_index < len(SUCCESS_THRESHOLDS):
        # 1. Initialize the training
        model.learn(total_timesteps=10000, reset_num_timesteps=False,
                    tb_log_name=run_name, callback=callbacks)
        total_timesteps += 10000
        
        # 2. Check the status of the training
        training_summary = training_monitor.get_training_summary()
        current_success_count = training_summary['current_success_count']
        total_episodes = training_summary['total_episodes']
        
        # 3. Update alpha based on the strategy
        new_alpha = wrapped_env.alpha

        if strategy == 1:
            # Continuous
            
            """
            TODO
            
            Make it pass the alpha parameter from the terminal command line
            or
            Manually change the alpha value
            """
            new_alpha = 1.0
            #new_alpha = 0.9
            #new_alpha = 0.8
            #new_alpha = 0.7
            #new_alpha = 0.6
            #new_alpha = 0.5

        elif strategy == 2:
            # Performance-Based

            """
            TODO

            Test new_alpha parameters & current_success_count
            Optimize based on the result from the strategy 1
            """
            if current_success_count > 800:
                new_alpha = 0.0
            elif current_success_count > 500:
                new_alpha = 0.1
            elif current_success_count > 200:
                new_alpha = 0.3
            else:
                new_alpha = 0.5

        elif strategy == 3:
            # Progressive

            """
            TODO

            Test max_steps or total steps traditional training requires
            Apply max_steps with a reasonable margin based on the result eg. if 90 -> 100
            """
            max_steps = 1_000_000
            frac = min(total_timesteps / max_steps, 1.0)
            new_alpha = 0.5 * (1.0 - frac)

        elif strategy == 4:
            # Need-Based

            """
            TODO

            Test and implement appropriate max alpha value
            """

            if len(training_monitor.success_counts) >= 2:
                prev_success = training_monitor.success_counts[-2]

                if current_success_count == prev_success:
                # 최근 success_count가 전과 동일 -> 성공이 늘어나지 않았음
                # ==> 학생이 학습 정체 상태로 추정 -> 코치 개입도(alpha) ↑
                    new_alpha += 0.05
                
                else:
                # success_count가 늘어났다면 -> 학습이 잘 진행됨
                # ==> 학생이 스스로 잘 하고 있으니 -> 코치 개입도(alpha) ↓
                    new_alpha -= 0.05


            new_alpha = max(0.0, min(0.5, new_alpha))

        # 4. Apply
        wrapped_env.set_alpha(new_alpha)
        wrapped_eval_env.set_alpha(new_alpha)

        # 5. Print training logs
        print(f"Training Progress:")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Total episodes: {training_summary['total_episodes']}")
        print(f"Current success count: {training_summary['current_success_count']}")
        print(f"Training time: {training_summary['training_time']}")
        print(f"Average reward: {training_summary['average_reward']:.2f}")
        print("-" * 50)

        # 6. Check if we've reached the success threshold
        if current_success_count >= SUCCESS_THRESHOLDS[threshold_index]:
            print(f"\nReached {SUCCESS_THRESHOLDS[threshold_index]} successes!")
            # Save checkpoint model
            model.save(os.path.join(checkpoint_dir, f"model_{SUCCESS_THRESHOLDS[threshold_index]}_successes"))
            threshold_index += 1

            # If we've reached all thresholds, break the loop
            if threshold_index >= len(SUCCESS_THRESHOLDS):
                end_time = time.time()
                training_time = end_time - start_time
                print(f"\nTraining completed! All success thresholds achieved.")
                print(f"Total training time: {training_time:.2f} seconds")
                print(f"Total episodes: {total_episodes}")
                break
    
    # Save final model and plot results
    model.save(os.path.join(models_dir, f"{run_name}_final"))

def test_coach_student(env, path_to_model):
    """
    Test the trained student model WITHOUT any coach involvement
    """

    # Load the trained student model to test
    model = SAC.load(path_to_model, env=env)

    obs, info = env.reset()
    terminated = truncated = False
    step_count = 0
    total_reward = 0

    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        step_count += 1
    
    print(f"\nTest ended after {step_count} steps, total reward={total_reward}")

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train or test coach-student model.')
    parser.add_argument('-train', '--train', action='store_true', help='Train with coach-student wrapper')
    parser.add_argument('-test', '--test', metavar='PATH', help='Path to trained model for testing')
    parser.add_argument('-coach', '--coach', metavar='PATH', default='reference_model.zip', help='Path to coach model')
    parser.add_argument('--strategy', type=int, default=1, help='Coaching strategy (1=Continuous, 2=Performance, 3=Progressive, 4=Need-Based)')
    args = parser.parse_args()

    # Create the unicycle environment
    env = gym.make('Student-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    # Now passing strategy options too
    if args.train:
        train_coach_student(env, coach_model_path=args.coach, strategy=args.strategy)

    if args.test:
        if os.path.isfile(args.test):
            test_env = gym.make('Student-v0', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            test_coach_student(test_env, path_to_model=args.test)
        else:
            print(f"{args.test} not found.")