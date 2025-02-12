from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time

class UnifiedMonitor(BaseCallback):
    """
    Unified monitoring system for:
      - Overall metrics (episodes, length, truncated/terminated, etc.)
      - Student agent metrics (rewards, actions, success rate, independence)
      - Coach-related metrics (alpha, intervention frequency, etc.)

    Provides real-time visualization through TensorBoard, keeping each component's
    metrics separate yet accessible in a single callback.
    """

    def __init__(self, verbose=0, window_size=100, intervention_threshold=0.1):
        super().__init__(verbose=verbose)
        self.window_size = window_size
        self.intervention_threshold = intervention_threshold  # Threshold for coach-student action difference

        # ------------------------- Overall monitoring -------------------------
        self.start_time = None               # Wall-clock time when training started                                                                     
        self.num_episodes = 0                # Total completed episodes
        self.current_ep_length = 0           # Current episode length (step count)
        self.episode_lengths = []            # History of episode lengths
        self.terminated_episodes = 0         # Count of episodes that ended normally
        self.truncated_episodes = 0          # Count of episodes truncated by time limit

        # ------------------------- Student monitoring -------------------------
        self.current_ep_reward = 0.0         # Accumulated reward in the current episode
        self.episode_rewards = []            # History of episode rewards
        self.episode_successes = []          # History of whether each episode reached the goal (1.0 or 0.0)
        self.independent_successes = 0       # Count of episodes with zero coach intervention + success

        # Optional: tracking student actions (size=3 for each step)
        self.current_ep_student_actions = [] # Actions in current episode
        self.episode_actions_stats = []      # (Optional) store per-episode action stats if needed

        # ------------------------- Coach monitoring -------------------------
        self.current_ep_intervention_count = 0  # Number of interventions in the current episode
        self.episode_interventions = []         # History of intervention counts per episode

        # Alpha tracking (optional)
        self.current_ep_alpha = []
        self.alpha_history = []

    def _on_training_start(self):
        """
        Called before the first training rollout.
        """
        self.start_time = time.time()

    def get_training_summary(self):
        """
        Provides a concise snapshot of the training progress.
        Used for strategy updates or logging in train_coach_student.
        """
        training_time_sec = time.time() - self.start_time if self.start_time else 0
        hours, remainder = divmod(training_time_sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        total_episodes = self.num_episodes
        current_success_count = int(np.sum(self.episode_successes))
        average_reward = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

        return {
            "total_episodes": total_episodes,
            "current_success_count": current_success_count,
            "average_reward": average_reward,
            "training_time": formatted_time,
        }

    def _on_step(self) -> bool:
        """
        Called every time 'env.step()' is executed. 
        We assume a single environment, so self.locals[...] will be length=1 lists.
        """
        # Retrieve SB3 vectors (usually length=1 for single env)
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]
        infos   = self.locals["infos"]

        reward = rewards[0]
        done   = dones[0]
        info   = infos[0]

        # (A) Update overall metrics each step
        self._update_overall_metrics(reward, done, info)

        # (B) Update student metrics each step
        self._update_student_metrics(reward, done, info)

        # (C) Update coach metrics each step
        self._update_coach_metrics()

        # If the episode ended, finalize logs for overall, student, and coach
        if done:
            self._log_overall_episode(info)
            self._log_student_episode(info)
            self._log_coach_episode(info)

        return True

    def _on_training_end(self):
        """
        Called when training finishes (model.learn() is done).
        """
        if self.verbose > 0:
            total_time = time.time() - self.start_time if self.start_time else 0
            print(f"[UnifiedMonitor] Training finished - Episodes: {self.num_episodes}, "
                  f"Elapsed time: {total_time:.2f} s")

    # --------------------------------------------------------------------------
    # (1) Overall metrics
    # --------------------------------------------------------------------------
    def _update_overall_metrics(self, reward, done, info):
        """
        Step-by-step updates for overall metrics (episode length, etc.).
        """
        self.current_ep_length += 1

    def _log_overall_episode(self, info):
        """
        Called when an episode finishes (done=True). Logs overall episode stats.
        """
        self.num_episodes += 1
        self.episode_lengths.append(self.current_ep_length)

        # Distinguish truncated vs. normal termination
        if info.get("TimeLimit.truncated", False):
            self.truncated_episodes += 1
        else:
            self.terminated_episodes += 1

        # Compute average episode length in the recent window
        recent_lengths = self.episode_lengths[-self.window_size:]
        mean_length = np.mean(recent_lengths)

        # Elapsed wall time
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        # Record to TensorBoard (under 'overall/')
        self.logger.record("overall/episode_length", self.current_ep_length)
        self.logger.record("overall/mean_length", mean_length)
        self.logger.record("overall/total_episodes", self.num_episodes)
        self.logger.record("overall/elapsed_time_sec", elapsed_time)
        self.logger.record("overall/terminated_episodes", self.terminated_episodes)
        self.logger.record("overall/truncated_episodes", self.truncated_episodes)
        self.logger.record("overall/total_timesteps", self.num_timesteps)

        # Reset for next episode
        self.current_ep_length = 0

    # --------------------------------------------------------------------------
    # (2) Student metrics
    # --------------------------------------------------------------------------
    def _update_student_metrics(self, reward, done, info):
        """
        Step-by-step updates for the student's performance, reward, actions, etc.
        """
        self.current_ep_reward += reward

        # Track student's actions (if using CoachStudentWrapper with last_student_action)
        wrapped_env = self.training_env.envs[0]
        if getattr(wrapped_env, "last_student_action", None) is not None:
            student_action = wrapped_env.last_student_action  # shape: (3,)
            self.current_ep_student_actions.append(student_action)

    def _log_student_episode(self, info):
        """
        Called when an episode finishes. Logs final student-related metrics
        (reward, success, action stats, etc.).
        """
        # 1) Reward & success
        self.episode_rewards.append(self.current_ep_reward)
        goal_reached = info.get("goal_reached", False)
        self.episode_successes.append(float(goal_reached))

        # 2) Rolling averages
        recent_rewards = self.episode_rewards[-self.window_size:]
        mean_reward = np.mean(recent_rewards)
        recent_successes = self.episode_successes[-self.window_size:]
        mean_success = np.mean(recent_successes)

        # Independent success: no coach intervention + goal reached
        if goal_reached and (self.current_ep_intervention_count == 0):
            self.independent_successes += 1

        # 3) Action distribution stats (optional)
        if len(self.current_ep_student_actions) > 0:
            actions_array = np.array(self.current_ep_student_actions)  # shape: (ep_steps, 3)
            mean_action = actions_array.mean(axis=0)
            std_action  = actions_array.std(axis=0)

            # Log each dimension's mean/std
            self.logger.record("student/action_wheel_mean", mean_action[0])
            self.logger.record("student/action_wheel_std",  std_action[0])
            self.logger.record("student/action_roll_mean",  mean_action[1])
            self.logger.record("student/action_roll_std",   std_action[1])
            self.logger.record("student/action_yaw_mean",   mean_action[2])
            self.logger.record("student/action_yaw_std",    std_action[2])

        # 4) Log final student scalars
        self.logger.record("student/episode_reward", self.current_ep_reward)
        self.logger.record("student/mean_reward", mean_reward)
        self.logger.record("student/mean_success", mean_success)
        self.logger.record("student/current_successes", np.sum(self.episode_successes))
        self.logger.record("student/independent_successes", self.independent_successes)

        # Reset for next episode
        self.current_ep_reward = 0.0
        self.current_ep_student_actions.clear()

    # --------------------------------------------------------------------------
    # (3) Coach metrics
    # --------------------------------------------------------------------------
    def _update_coach_metrics(self):
        """
        Step-by-step updates for coach-related metrics (alpha, interventions, etc.).
        """
        wrapped_env = self.training_env.envs[0]

        # 1) Track alpha (if present)
        if hasattr(wrapped_env, "alpha"):
            alpha_val = wrapped_env.alpha
            self.current_ep_alpha.append(alpha_val)

        # 2) Compare coach vs. student actions for interventions
        if (getattr(wrapped_env, "last_coach_action", None) is not None and
            getattr(wrapped_env, "last_student_action", None) is not None):
            coach_action = wrapped_env.last_coach_action
            student_action = wrapped_env.last_student_action
            difference = np.linalg.norm(coach_action - student_action)
            # If difference > threshold, we count an intervention
            if difference > self.intervention_threshold:
                self.current_ep_intervention_count += 1

    def _log_coach_episode(self, info):
        """
        Called when an episode finishes. Logs coach-related metrics (alpha stats, intervention count, etc.).
        """
        # Alpha stats
        if len(self.current_ep_alpha) > 0:
            alpha_array = np.array(self.current_ep_alpha)
            mean_alpha = alpha_array.mean()
            std_alpha  = alpha_array.std()
            self.logger.record("coach/mean_alpha", mean_alpha)
            self.logger.record("coach/std_alpha",  std_alpha)
            # Keep entire history if needed
            self.alpha_history.extend(self.current_ep_alpha)

        # Interventions this episode
        self.episode_interventions.append(self.current_ep_intervention_count)
        self.logger.record("coach/intervention_count", self.current_ep_intervention_count)

        # Reset coach counters for next episode
        self.current_ep_alpha.clear()
        self.current_ep_intervention_count = 0
