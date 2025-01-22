from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CoachMonitorCallback(BaseCallback):
    """
    Custom callback for monitoring coach behavior during training.
    Tracks three main metrics:
    1. alpha_value: degree of coach intervention
    2. intervention_frequency: how often coach actually intervenes
    3. contribution: impact of coach's actions on final behavior
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Initialize counters for tracking coach interventions
        self.intervention_counter = 0
        self.total_steps = 0
    
    def _on_step(self) -> bool:
        try:
            self.total_steps += 1
            
            # Access the actual wrapper environment from vectorized env
            # training_env.envs[0] gets the first env from VecEnv
            wrapped_env = self.training_env.envs[0]
            
            # Log current alpha value (intervention degree)
            alpha = wrapped_env.alpha
            self.logger.record("coach/alpha_value", alpha)
            
            # Retrieve stored actions from wrapper
            coach_action = wrapped_env.last_coach_action
            student_action = wrapped_env.last_student_action
            
            if coach_action is not None and student_action is not None:
                # Calculate intervention frequency
                # Consider it an intervention if coach action is non-zero
                if np.any(np.abs(coach_action) > 0.01):  # Using small threshold
                    self.intervention_counter += 1
                intervention_freq = self.intervention_counter / self.total_steps
                self.logger.record("coach/intervention_frequency", intervention_freq)
                
                # Calculate coach's contribution
                # Using Euclidean distance between coach and student actions
                contribution = np.linalg.norm(coach_action - student_action)
                self.logger.record("coach/contribution", contribution)
            
            return True
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Failed to log coach information: {str(e)}")
            return True