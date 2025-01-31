from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time
import os

class UnifiedMonitor(BaseCallback):
    """
    Unified monitoring system that tracks both student and coach interactions.
    This system provides real-time visualization through TensorBoard while maintaining
    the independence of each component's metrics.
    
    Key features:
    1. Student learning progress tracking
    2. Coach intervention monitoring
    3. Overall training progress tracking
    4. Real-time metric visualization
    """
    def __init__(self, run_dir, evaluation_window=100, independence_threshold=0.1, verbose=0):
        super().__init__(verbose)
        # System configuration
        self.run_dir = run_dir
        self.setup_directories()
        self.training_start_time = time.time()
        self.evaluation_window = evaluation_window
        self.independence_threshold = independence_threshold

        # Student-related metrics initialization
        self.student_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_count': 0,
            'independent_success_count': 0,
            'current_episode_reward': 0,
            'previous_success_rate': 0.0
        }

        # Coach-related metrics initialization
        self.coach_metrics = {
            'intervention_counter': 0,
            'total_steps': 0,
            'episode_interventions': [],
            'current_episode_interventions': 0
        }

        # Overall training metrics initialization
        self.training_metrics = {
            'success_counts': [],
            'episode_counts': [],
            'training_times': [],
            'cumulative_rewards': []
        }

    def setup_directories(self):
        """Sets up directory structure for storing monitoring data."""
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

    def _on_step(self) -> bool:
        try:
            # Collect basic step information
            wrapped_env = self.training_env.envs[0]
            info = self.locals['infos'][0]
            done = self.locals['dones'][0]
            reward = self.locals['rewards'][0]

            # Update metrics for each component
            self._update_student_metrics(reward, info, done)
            self._update_coach_metrics(wrapped_env)
            self._update_training_metrics(info, reward)

            return True

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Failed to log metrics: {str(e)}")
            return True

    def _update_student_metrics(self, reward, info, done):
        """
        Updates student learning progress metrics.
        Tracks rewards, success rates, and independence measures.
        """
        self.student_metrics['current_episode_reward'] += reward

        if done:
            # Process episode completion metrics
            if info.get('success', False):
                self.student_metrics['success_count'] += 1
                if self.coach_metrics['current_episode_interventions'] == 0:
                    self.student_metrics['independent_success_count'] += 1

            # Calculate and log performance metrics
            self._log_performance_metrics()
            
            # Reset episode counters
            self.student_metrics['current_episode_reward'] = 0

    def _update_coach_metrics(self, wrapped_env):
        """
        Updates coach intervention metrics.
        Monitors intervention frequency and impact on student actions.
        """
        self.coach_metrics['total_steps'] += 1
        
        if wrapped_env.last_coach_action is not None:
            # Check for coach intervention
            if np.any(np.abs(wrapped_env.last_coach_action) > self.independence_threshold):
                self.coach_metrics['current_episode_interventions'] += 1
                self.coach_metrics['intervention_counter'] += 1

            # Log coach-related metrics
            self._log_coach_metrics(wrapped_env)

    def _update_training_metrics(self, info, reward):
        """
        Updates overall training progress metrics.
        Tracks episode counts, success counts, and cumulative rewards.
        """
        # Update success counts if available
        if 'success_count' in info:
            self.training_metrics['success_counts'].append(info['success_count'])
        
        # Update episode counts when episode completes
        if info.get('terminal_observation') is not None:
            self.training_metrics['episode_counts'].append(
                len(self.training_metrics['episode_counts']) + 1
            )
        
        # Update training times
        current_time = time.time() - self.training_start_time
        self.training_metrics['training_times'].append(current_time)
        
        # Update cumulative rewards
        if self.training_metrics['cumulative_rewards']:
            self.training_metrics['cumulative_rewards'].append(
                self.training_metrics['cumulative_rewards'][-1] + reward
            )
        else:
            self.training_metrics['cumulative_rewards'].append(reward)

    def _log_performance_metrics(self):
        """
        Records performance metrics to TensorBoard.
        Includes pure performance and independence rate calculations.
        """
        if len(self.student_metrics['episode_rewards']) >= self.evaluation_window:
            recent_performance = np.mean(
                self.student_metrics['episode_rewards'][-self.evaluation_window:]
            )
            self.logger.record("student/pure_performance", recent_performance)

        if self.student_metrics['success_count'] > 0:
            independence_rate = (self.student_metrics['independent_success_count'] / 
                               self.student_metrics['success_count'])
            self.logger.record("student/independence_rate", independence_rate)

    def _log_coach_metrics(self, wrapped_env):
        """
        Records coach-related metrics to TensorBoard.
        Tracks intervention frequency and contribution level.
        """
        intervention_freq = (self.coach_metrics['intervention_counter'] / 
                           self.coach_metrics['total_steps'])
        self.logger.record("coach/intervention_frequency", intervention_freq)
        
        if wrapped_env.last_student_action is not None:
            contribution = np.linalg.norm(
                wrapped_env.last_coach_action - wrapped_env.last_student_action
            )
            self.logger.record("coach/contribution", contribution)

    def get_training_summary(self):
        """
        Provides a comprehensive summary of current training progress.
        Returns key metrics including success rate, independence rate, and training duration.
        """
        training_time = time.time() - self.training_start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_format = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        return {
            'total_episodes': len(self.training_metrics['episode_counts']),
            'current_success_count': self.student_metrics['success_count'],
            'independence_rate': (self.student_metrics['independent_success_count'] / 
                                max(1, self.student_metrics['success_count'])),
            'average_reward': np.mean(self.student_metrics['episode_rewards'] or [0]),
            'training_time': time_format,
            'intervention_rate': (self.coach_metrics['intervention_counter'] / 
                                max(1, self.coach_metrics['total_steps']))
        }