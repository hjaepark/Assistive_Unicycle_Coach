from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class StudentMonitorCallback(BaseCallback):
    """
    Custom callback for monitoring student agent's performance metrics.
    Tracks three key aspects of student learning:
    1. Pure performance (reward-based evaluation)
    2. Learning progress (improvement over time)
    3. Independence rate (success without coach intervention)
    """
    def __init__(self, verbose=0, evaluation_window=100, independence_threshold=0.1):
        super().__init__(verbose)
        # Initialize basic tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
        
        # Window size for rolling metrics calculation
        self.evaluation_window = evaluation_window
        
        # Independence tracking variables
        self.independence_threshold = independence_threshold
        self.independent_success_count = 0
        self.episode_interventions = []
        
        # Learning progress tracking
        self.previous_success_rate = 0.0
        self.success_history = []
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_interventions = 0
    
    def _on_step(self) -> bool:
        try:
            # Access wrapped environment and current step information
            wrapped_env = self.training_env.envs[0]
            info = self.locals['infos'][0]
            done = self.locals['dones'][0]
            reward = self.locals['rewards'][0]
            
            # Accumulate current episode data
            self.current_episode_reward += reward
            
            # Track coach interventions
            if wrapped_env.last_coach_action is not None:
                if np.any(np.abs(wrapped_env.last_coach_action) > self.independence_threshold):
                    self.current_episode_interventions += 1
            
            if done:
                self.total_episodes += 1
                # Store episode data
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_interventions.append(self.current_episode_interventions)
                
                # Track successful episodes
                if info.get('success', False):
                    self.success_count += 1
                    # Consider it independent if interventions are minimal
                    if self.current_episode_interventions == 0:
                        self.independent_success_count += 1
                
                # Calculate and log pure performance
                if len(self.episode_rewards) >= self.evaluation_window:
                    recent_performance = np.mean(self.episode_rewards[-self.evaluation_window:])
                    self.logger.record("student/pure_performance", recent_performance)
                
                # Calculate and log learning progress
                if self.total_episodes % self.evaluation_window == 0:
                    current_success_rate = self.success_count / self.total_episodes
                    learning_progress = current_success_rate - self.previous_success_rate
                    self.previous_success_rate = current_success_rate
                    self.logger.record("student/learning_progress", learning_progress)
                
                # Calculate and log independence metrics
                if self.success_count > 0:
                    independence_rate = self.independent_success_count / self.success_count
                    self.logger.record("student/independence_rate", independence_rate)
                    
                    # Log additional independence metrics
                    avg_interventions = np.mean(self.episode_interventions[-self.evaluation_window:])
                    self.logger.record("student/average_interventions", avg_interventions)
                
                # Reset episode-specific counters
                self.current_episode_reward = 0
                self.current_episode_interventions = 0
            
            return True
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Failed to log student metrics: {str(e)}\n"
                      f"Error occurred at episode {self.total_episodes}")
            return True