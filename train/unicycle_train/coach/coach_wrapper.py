# train/unicycle_train/coach/coach_wrapper.py

import gymnasium as gym
import numpy as np

from train.unicycle_train.coach.coach_buffer import CoachReplayBuffer

class CoachWrapper(gym.Wrapper):
    def __init__(self, env, coach_agent=None, estimator=None):
        super().__init__(env)
        self.coach_agent = coach_agent
        self.estimator = estimator
        self.coach_buffer = CoachReplayBuffer(capacity=100_000) if coach_agent else None
        
        # weight_assist parameters
        # TODO & kappa is temporary fix for pipeline test
        # (continuous 0~1; no constraints yet)
        self.kappa = np.array([0.3, 0.3], dtype=np.float32)
        # initial skill level of the student agent before the estimator updates it
        self.skill_level = 0.0

        # Track internal state
        self._last_obs = None
        self.episode_return = 0.0
        self.episode_coach_rewards = []

        # Track rollout statistics
        self.rollout_weight_assists = []
        self.rollout_coach_action_norms = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self.episode_return = 0.0
        self.episode_coach_rewards = []

        # Reset rollout statistics
        self.rollout_weight_assists = []
        self.rollout_coach_action_norms = []

        return obs, info

    def step(self, student_action):
        # 0. Pass through if no coach is provided
        if self.coach_agent is None:
            print ("[CoachWrapper] No coach provided, pass-through.")
            return self.env.step(student_action)

        # 1) Construct coach observation 20D (19D + skill 1D)
        coach_obs = np.concatenate([self._last_obs, [self.skill_level]])
        
        # 2) Predict coach action + weight_assist
        coach_action, weight_assist = self.coach_agent.predict(coach_obs, deterministic=False)
        weight_assist = np.clip(weight_assist, 0.0, 1.0)
        
        # 4) Merge actions
        actual_coach_action = weight_assist * self.kappa * coach_action
        exec_action = np.clip(student_action + actual_coach_action, -1.0, 1.0)

        # 5) Run environment step
        next_obs, reward, terminated, truncated, info = self.env.step(exec_action)
        
        # 6) Compute coach reward
        coach_reward = self._compute_coach_reward(info, weight_assist, coach_action)
        self.episode_coach_rewards.append(coach_reward)

        self.rollout_weight_assists.append(weight_assist)
        self.rollout_coach_action_norms.append(np.linalg.norm(self.kappa * coach_action))
        
        # 7) Save to coach buffer
        if self.coach_buffer is not None:
            next_coach_obs = np.concatenate([next_obs, [self.skill_level]])
            self.coach_buffer.add(
                coach_obs, coach_action, coach_reward,
                next_coach_obs, terminated or truncated, weight_assist
            )
        
        # 8) Update skill estimator at episode end
        if (terminated or truncated) and self.estimator:
            if "episode" in info and "r" in info["episode"]:
                self.skill_level = self.estimator.update(info["episode"]["r"])
                print(f"[CoachWrapper] Updated skill level to {self.skill_level:.3f}")
        
        # 9) Extend info dict
        info["weight_assist"] = weight_assist
        info["coach_action_norm"] = np.linalg.norm(self.kappa * coach_action)
        info["skill_level"] = self.skill_level

        # Add epsisode-level rollout statistics at episode end
        if terminated or truncated:
            info["rollout/coach_reward_mean"] = float(np.mean(self.episode_coach_rewards))
            info["rollout/coach_reward_sum"] = float(np.sum(self.episode_coach_rewards))
            info["rollout/weight_assist_mean"] = float(np.mean(self.rollout_weight_assists))
            info["rollout/weight_assist_std"] = float(np.std(self.rollout_weight_assists))
            info["rollout/coach_action_norm_mean"] = float(np.mean(self.rollout_coach_action_norms))
            info["rollout/skill_level"] = float(self.skill_level)        
        
        self._last_obs = next_obs
        self.episode_return += reward
        
        return next_obs, reward, terminated, truncated, info
    

    # TODO: compute weight_assist part needs to be further developed for constraints

        
    def _compute_coach_reward(self, info, weight_assist, coach_action):
        # Coach reward architecture

        # 1) Safety Reward: Encourage upright posture, penalize falls
        upright_gate = info.get("upright_gate", 1.0)
        is_fallen = info.get("is_fallen", False)
        goal_reached = info.get("goal_reached", False)
        
        safety_reward = 0.5 * upright_gate
        if is_fallen:
            safety_reward -= 10.0
        
        # 2) Task Reward: Encourage reaching the goal
        task_reward = 10.0 if goal_reached else 0.0
        
        # 3) Assistance Penalty: Penalize excessive assistance
        assist_penalty = -0.1 * np.linalg.norm(self.kappa * coach_action) ** 2
        
        # 4) Progress Reward: Small reward for making progress
        progress_reward = 0.01 * self.episode_return
        
        # Total Coach Reward
        total_reward = safety_reward + task_reward + assist_penalty + progress_reward
        
        return float(total_reward)
    
    def update_coach(self, n_gradient_steps=10, learning_starts=1000):
        """Update coach agent and return comprehensive logs."""
        if self.coach_agent and self.coach_buffer and self.coach_buffer.size > learning_starts:
            # This now includes both train/ and buffer/ metrics
            logs = self.coach_agent.update_from_buffer(
                self.coach_buffer, 
                n_gradient_steps=n_gradient_steps
            )
            return logs
        return {}