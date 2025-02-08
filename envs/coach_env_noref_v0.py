import gym
import numpy as np

class CoachEnv(gym.Env):
    """
    A custom environment for training the coach agent.
    The observation and action spaces are taken from the student environment.
    The coach’s reward is computed from the student’s learning progress.
    
    Reward mode:
      1: Use an exponentially weighted moving average (EWMA) of the student's reward differences.
      2: (Mode 1) + add linear regression slope over the last 10 timesteps.
      3: (Mode 1) + (Mode 2) + add linear regression slope over the last 100 timesteps.
      4: (Mode 1) + (Mode 2) + (Mode 3) + add linear regression slope over the last 1000 timesteps.
    """
    def __init__(self, student_env, reward_mode=1):
        super().__init__()
        self.student_env = student_env
        self.reward_mode = reward_mode

        # Use the same observation and action spaces as the student.
        self.observation_space = student_env.observation_space
        self.action_space = student_env.action_space

        # To track the student's reward history.
        self.student_rewards = []

    def reset(self, **kwargs):
        self.student_rewards = []
        obs, _ = self.student_env.reset(**kwargs)
        return obs

    def step(self, action):
        # Coach's action is merged with the student's via the CoachStudentWrapper so that calling step(action) on the
        # student environment returns the next observation and student's reward.
        obs, student_reward, done, truncated, info = self.student_env.step(action)
        self.student_rewards.append(student_reward)
        
        # Compute the coach's reward from the student's reward history.
        coach_reward = self._compute_coach_reward()
        return obs, coach_reward, done, truncated, info

    def _compute_coach_reward(self):
        # If not enough data, return 0 reward.
        if len(self.student_rewards) < 2:
            return 0.0

        # Compute reward differences between consecutive timesteps.
        diffs = np.diff(self.student_rewards)

        # Experiment 1: Compute an EWMA of the differences.
        smoothing = 0.9
        ewma = 0.0
        for d in diffs:
            ewma = smoothing * ewma + (1 - smoothing) * d
        coach_reward = ewma

        # For reward modes 2, 3, and 4 add a linear regression slope term:
        if self.reward_mode in [2, 3, 4]:
            if self.reward_mode == 2:
                N = 10
            elif self.reward_mode == 3:
                N = 100
            elif self.reward_mode == 4:
                N = 1000
            if len(diffs) >= N:
                x = np.arange(N)
                y = np.array(diffs[-N:])
                # Simple linear regression to obtain the slope:
                A = np.vstack([x, np.ones(len(x))]).T
                slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]

                #TODO: need to provide weight adjustment for the reward for slope
                coach_reward += slope

        return coach_reward