# train/unicycle_train/coach/v0/wrapper.py

import gymnasium as gym
import numpy as np
from train.unicycle_train.coach.v0.coach_buffer import CoachReplayBuffer


class CoachWrapper(gym.Wrapper):
    """
    Wrapper that adds frozen coach assist to student's actions
    and manages coach buffer storage during inner loop.
    
    This wrapper is used DURING the inner loop only.
    The outer loop logic (coach update, student reset) is in train.py.
    """

    def __init__(self, env, coach_agent, coach_buffer: CoachReplayBuffer):
        super().__init__(env)
        self.coach_agent = coach_agent
        self.coach_buffer = coach_buffer

        # Internal state
        self._last_obs = None
        self._last_reward = 0.0

        # Inner loop tracking
        self.inner_loop_steps = 0
        self.inner_loop_sim_time = 0.0
        self.consecutive_successes = 0

        # Episode tracking
        self.episode_count = 0
        self.total_episodes = 0

        self._pending_transition = None

    def reset(self, **kwargs):
        """Reset environment for a new episode within inner loop."""
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_reward = 0.0
        return obs, info

    def reset_inner_loop(self):
        """
        Reset all inner loop state for a new student.
        Called at the start of each outer loop iteration.
        """
        self.inner_loop_steps = 0
        self.inner_loop_sim_time = 0.0
        self.consecutive_successes = 0
        self.episode_count = 0
        self._pending_transition = None

    def step(self, student_action):
        # Construct coach observation: (s_t, a^s_t, r_env)
        coach_obs = np.concatenate([
            self._last_obs,
            student_action,
            [self._last_reward],
        ]).astype(np.float32)

        # Get coach assist action (frozen policy)
        coach_action = self.coach_agent.predict(coach_obs, deterministic=False)

        # Combine actions
        combined_action = np.clip(student_action + coach_action, -1.0, 1.0)

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(combined_action)

        # Commit previous pending transition
        if self._pending_transition is not None:
            self.coach_buffer.add(
                obs=self._pending_transition["coach_obs"],
                action=self._pending_transition["coach_action"],
                reward=0.0,
                next_obs=coach_obs,
                done=self._pending_transition["done"],
            )

        # Store current as pending (will be completed next step)
        done = bool(terminated or truncated)
        self._pending_transition = {
            "coach_obs": coach_obs,
            "coach_action": coach_action,
            "done": done,
        }

        # If episode ends, commit immediately with terminal next_obs
        if done:
            terminal_next_obs = np.concatenate([
                obs,
                np.zeros(2),       # No next student action at terminal
                [reward],
            ]).astype(np.float32)

            self.coach_buffer.add(
                obs=coach_obs,
                action=coach_action,
                reward=0.0,
                next_obs=terminal_next_obs,
                done=True,
            )
            self._pending_transition = None

        # Update tracking
        self.inner_loop_steps += 1
        self.inner_loop_sim_time += self.env.unwrapped.dt
        self._last_obs = obs
        self._last_reward = reward

        # Track episode success/failure
        if done:
            self.episode_count += 1
            self.total_episodes += 1

            if info.get("goal_reached", False):
                self.consecutive_successes += 1
            else:
                self.consecutive_successes = 0

        info["coach_action"] = coach_action
        info["combined_action"] = combined_action
        info["inner_loop_steps"] = self.inner_loop_steps
        info["inner_loop_sim_time"] = self.inner_loop_sim_time
        info["consecutive_successes"] = self.consecutive_successes
        info["inner_loop_episode"] = self.episode_count

        return obs, reward, terminated, truncated, info

    def is_converged(self, n_consecutive=3):
        """Check if student has converged (N consecutive successes)."""
        return self.consecutive_successes >= n_consecutive

    def is_timed_out(self, sim_time_max):
        """Check if inner loop has exceeded maximum simulation time."""
        return self.inner_loop_sim_time >= sim_time_max

    def get_inner_loop_summary(self):
        """Return summary of the completed inner loop for logging."""
        return {
            "inner_loop/steps": self.inner_loop_steps,
            "inner_loop/sim_time": self.inner_loop_sim_time,
            "inner_loop/episodes": self.episode_count,
            "inner_loop/consecutive_successes": self.consecutive_successes,
            "inner_loop/converged": self.consecutive_successes >= 3,
        }