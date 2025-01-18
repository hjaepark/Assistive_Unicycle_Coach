# train/monitoring/coach_student_monitor.py
from stable_baselines3.common.callbacks import BaseCallback
import time
import numpy as np
import os

class CoachStudentMonitor(BaseCallback):
    """
    A monitoring class for the coach-student learning system.
    Responsible for tracking and storing key metrics throughout the learning process.
    """
    def __init__(self, run_dir, verbose=0):
        super(CoachStudentMonitor, self).__init__(verbose)
        self.setup_directories(run_dir)
        self.initialize_metrics()
        
    def setup_directories(self, run_dir):
        """Sets up the directory structure for storing monitoring data."""
        self.run_dir = run_dir
        self.metrics_dir = os.path.join(run_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def initialize_metrics(self):
        """Initializes metrics to be tracked during training."""
        # Success and episode metrics
        self.success_counts = []       # Cumulative success count
        self.episode_counts = []       # Episode count tracking
        self.episode_rewards = []      # Rewards per episode
        self.current_episode_reward = 0
        
        # Time-related metrics
        self.training_start_time = time.time()
        self.training_times = []
        
        # Learning progress metrics
        self.step_rewards = []         # Rewards per step
        self.cumulative_rewards = []   # Accumulated rewards
        
    def _on_step(self) -> bool:
        """Collects and processes data at each step."""
        # Collect current step information
        info = self.locals['infos'][0]
        reward = info.get('reward', 0)
        
        # Update basic metrics
        self.current_episode_reward += reward
        self.step_rewards.append(reward)
        self.success_counts.append(info['success_count'])
        self.training_times.append(time.time() - self.training_start_time)
        
        # Handle episode completion
        if info.get('terminal_observation') is not None:
            self.episode_counts.append(len(self.episode_counts) + 1)
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
            # Calculate cumulative rewards
            if self.cumulative_rewards:
                self.cumulative_rewards.append(
                    self.cumulative_rewards[-1] + reward)
            else:
                self.cumulative_rewards.append(reward)
        
        return True
    
    def get_training_summary(self):
        """Returns a summary of current training progress."""
        current_success = self.success_counts[-1] if self.success_counts else 0
        current_time = self.training_times[-1] if self.training_times else 0
        
        # Convert time format
        hours = int(current_time // 3600)
        minutes = int((current_time % 3600) // 60)
        seconds = int(current_time % 60)
        time_format = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return {
            'total_episodes': len(self.episode_counts),
            'current_success_count': current_success,
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'training_time': time_format,
            'total_timesteps': len(self.step_rewards)
        }