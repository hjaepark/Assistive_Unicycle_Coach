import gymnasium as gym
from stable_baselines3 import SAC
from gymnasium import spaces
import numpy as np

class CoachStudentWrapper(gym.Wrapper):
    """
    This wrapper merges actions from a coach policy and a student policy.
    The coach is trained via SAC on a custom environment (CoachEnv) that computes
    its reward based on the student's learning progress. 
    """

    # Temporary
    DEFAULT_MODE = "no-ref" # "no-ref": no reference model mode
    DEFAULT_ALPHA = 0.5 # Degree of coach action influence or intervention level
    DEFAULT_REWARD_COEF = 0.01 # Scaling factor (weight) applied during reward shaping
    DEFAULT_REWARD_SHAPING = False

    def __init__(
        self,
        env: gym.Env,
        coach_model_path: str = None, # If None, a new coach model is created.
        coach_reward_mode: int = 1, # 1, 2, 3, or 4 (experiment selection)
        mode: str = DEFAULT_MODE,
        alpha: float = DEFAULT_ALPHA,
        reward_coef: float = DEFAULT_REWARD_COEF,
        reward_shaping: bool = DEFAULT_REWARD_SHAPING
    ):

        super().__init__(env)
        self.mode = mode
        self.alpha = alpha
        self.reward_coef = reward_coef
        self.reward_shaping = reward_shaping

        # Create or load the coach model
        if coach_model_path is None:
            # Create a new coach environment that wraps the student environment
            self.coach_env = CoachEnv(student_env=env, reward_mode = coach_reward_mode)
            # Initialize a new SAC model for the coach
            self.coach_model = SAC("MlpPolicy", self.coach_env, verbose=1)
        else:
            # With reference method: load a pre-trained coach model
            self.coach_model = SAC.load(coach_model_path, env=None)

        # Add new attributes to store last actions
        self.last_coach_action = None
        self.last_student_action = None

        # Exisiting observation/action space
        self.observation_space = env.observation_space
        self.action_space = env.action_space  # Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def step(self, student_action):
        # 1. Use the current observation state stored during reset or previous step
        if not hasattr(self, '_current_obs'):
            raise RuntimeError("Environment needs to be reset before calling step")
        # Store student action
        self.last_student_action = student_action
        

        # 2. Get the coach's action from the RL-trained coach model
        coach_action, _ = self.coach_model.predict(self._current_obs, deterministic=True)
        # Store coach action
        self.last_coach_action = coach_action


        # 3. Merge student and coach action
        final_action = self._merge_actions(student_action, coach_action)

        # 4. Apply final action to the environment
        next_obs, reward, done, truncated, info = self.env.step(final_action)

        # 5. Store the next observation for future use
        self._current_obs = next_obs

        # (옵션) Reward shaping 등 보상 조정
        reward = self._modify_reward(reward, student_action, coach_action, done)

        return next_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        # Reset the action tracking attributes
        self.last_coach_action = None
        self.last_student_action = None
        
        # Initialize the environment and get the initial observation state
        # This stores the starting state needed for coach-student interaction
        self._current_obs, info = super().reset(**kwargs)
        return self._current_obs, info

    def _merge_actions(self, student_action: np.ndarray, coach_action: np.ndarray) -> np.ndarray:
        """
        Method of merging student and coach action depends on mode
        """

        # Mixture Policy (linear interpolation)
        if self.mode == "mixture":
            final_action = (1 - self.alpha) * student_action + self.alpha * coach_action

        # Add-on Policy
        elif self.mode == "add-on":
            final_action = student_action + self.alpha * (coach_action - student_action)

        # 100% coach action
        elif self.mode == "coach_only":
            final_action = coach_action

        # 100# student action
        elif self.mode == "student_only":
            final_action = student_action

        # No reference model just combines student and coach action
        elif self.mode == "no-ref":
            final_action = student_action + coach_action

        return final_action

    def _modify_reward(self, original_reward, student_action, coach_action, done):
        if not self.reward_shaping:  # reward_shaping이 False면
            return original_reward   # 원래 보상을 그대로 반환
        
        reward = original_reward
        if not done:
            if self.mode == "reward_shaping" or self.mode == "hybrid":
                similarity = -np.linalg.norm(coach_action - student_action)
                reward += self.reward_coef * similarity
        
        return reward
    
    def set_alpha(self, new_alpha: float):
        """
        Dynamically update the alpha value (coach influence).
        Clamps the alpha value to [0.0, 1.0].
        """
        self.alpha = np.clip(new_alpha, 0.0, 1.0)