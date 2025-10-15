# train/unicycle_train/coach/coach_wrapper.py

import gymnasium as gym
import numpy as np
from train.unicycle_train.coach.coach_buffer import CoachReplayBuffer
from train.unicycle_train.coach.coach_config import CoachRewardConfig

class CoachWrapper(gym.Wrapper):
    def __init__(self, env, coach_agent=None, estimator=None, 
                 reward_config: CoachRewardConfig = None):
        super().__init__(env)
        self.coach_agent = coach_agent
        self.estimator = estimator
        self.coach_buffer = CoachReplayBuffer(capacity=100_000) if coach_agent else None
        
        # Reward configuration
        self.reward_cfg = reward_config or CoachRewardConfig()
        self.kappa = self.reward_cfg.kappa
        
        # Skill level
        self.skill_level = 0.0
        
        # Track internal state
        self._last_obs = None
        self.episode_return = 0.0
        self.episode_coach_rewards = []
        
        # Rollout statistics
        self.rollout_weight_assists = []
        self.rollout_effective_intervention_norms = []
        
        # Responsibility tracking (EMA of weight_assist)
        self.resp_ema = 0.0
        
        # Previous values for delta calculation
        self.prev_upright = None
        self.prev_progress = None
        
        # Pending transition for correct MDP (s' includes a_{t+1})
        self._pending_transition = None  # ← 추가!

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self.episode_return = 0.0
        self.episode_coach_rewards = []
        self.rollout_weight_assists = []
        self.rollout_effective_intervention_norms = []
        self.resp_ema = 0.0
        self.prev_upright = None
        self.prev_progress = None
        self._pending_transition = None  # ← 추가!
        return obs, info

    def step(self, student_action):
        # 0. Pass through if no coach
        if self.coach_agent is None:
            return self.env.step(student_action)
        
        # 1) Commit previous transition if exists (now we have a_{t+1}^S)
        if self.coach_buffer is not None and self._pending_transition is not None:
            # Complete previous transition with current student_action as a_{t+1}^S
            next_coach_obs = np.concatenate([
                self._last_obs,           # o_{t+1} from previous step
                [self.skill_level],       # skill_{t+1}
                student_action            # a_{t+1}^S ← Correct next action!
            ])
            
            self.coach_buffer.add(
                self._pending_transition["coach_obs"],
                self._pending_transition["coach_action"],
                self._pending_transition["coach_reward"],
                next_coach_obs,  # ← Now correct s_{t+1}!
                self._pending_transition["done"],
                self._pending_transition["weight_assist"]
            )
            self._pending_transition = None
        
        # 2) Construct current coach observation: s_t = (o_t, skill_t, a_t^S)
        coach_obs = np.concatenate([
            self._last_obs,
            [self.skill_level],
            student_action
        ])
        
        # 3) Predict coach action + weight_assist
        coach_action, weight_assist = self.coach_agent.predict(coach_obs, deterministic=False)
        weight_assist = np.clip(weight_assist, 0.0, 1.0)
        
        # 4) Update responsibility EMA
        alpha = self.reward_cfg.resp_ema_alpha
        self.resp_ema = alpha * weight_assist + (1 - alpha) * self.resp_ema
        
        # 5) Compute interventions
        intended_intervention = weight_assist * self.kappa * coach_action
        exec_action = np.clip(student_action + intended_intervention, -1.0, 1.0)
        effective_intervention = exec_action - student_action
        
        # 6) Run environment step
        next_obs, reward, terminated, truncated, info = self.env.step(exec_action)
        
        # 7) Compute coach reward
        coach_reward = self._compute_coach_reward(
            coach_action=coach_action,
            weight_assist=weight_assist,
            effective_intervention=effective_intervention,
            info=info
        )
        self.episode_coach_rewards.append(coach_reward)
        
        # 8) Track statistics
        self.rollout_weight_assists.append(weight_assist)
        self.rollout_effective_intervention_norms.append(np.linalg.norm(effective_intervention))
        
        # 9) Store current transition as pending (waiting for a_{t+1}^S)
        if self.coach_buffer is not None:
            self._pending_transition = {
                "coach_obs": coach_obs,
                "coach_action": coach_action,
                "coach_reward": coach_reward,
                "done": bool(terminated or truncated),
                "weight_assist": weight_assist
            }
            
            # If episode ends, commit immediately with dummy next action
            if terminated or truncated:
                dummy_next_action = np.zeros_like(student_action)
                next_coach_obs_terminal = np.concatenate([
                    next_obs,
                    [self.skill_level],
                    dummy_next_action  # Dummy since episode ended
                ])
                
                self.coach_buffer.add(
                    coach_obs,
                    coach_action,
                    coach_reward,
                    next_coach_obs_terminal,
                    True,  # done=True
                    weight_assist
                )
                self._pending_transition = None
        
        # 10) Update skill estimator at episode end
        if (terminated or truncated) and self.estimator:
            if "episode" in info and "r" in info["episode"]:
                self.skill_level = self.estimator.update(info["episode"]["r"])
        
        # 11) Extend info
        info["weight_assist"] = weight_assist
        info["effective_intervention_norm"] = float(np.linalg.norm(effective_intervention))
        info["skill_level"] = self.skill_level
        info["resp_ema"] = self.resp_ema
        
        # Add episode-level rollout statistics
        if terminated or truncated:
            info["rollout/coach_reward_mean"] = float(np.mean(self.episode_coach_rewards))
            info["rollout/coach_reward_sum"] = float(np.sum(self.episode_coach_rewards))
            info["rollout/weight_assist_mean"] = float(np.mean(self.rollout_weight_assists))
            info["rollout/weight_assist_std"] = float(np.std(self.rollout_weight_assists))
            info["rollout/effective_intervention_norm_mean"] = float(np.mean(self.rollout_effective_intervention_norms))
            info["rollout/skill_level"] = float(self.skill_level)
        
        self._last_obs = next_obs
        self.episode_return += reward
        
        return next_obs, reward, terminated, truncated, info
    
    def _compute_coach_reward(self, coach_action, weight_assist, effective_intervention, info):
        """
        Modular coach reward: (s, a, s') form only (Markov)
        Uses info from environment (which contains t+1 state info)
        """
        cfg = self.reward_cfg
        total_reward = 0.0
        
        # Get current values from info (t+1 state)
        upright = float(info.get("upright_gate", 1.0))
        progress = float(info.get("reward_progress_gated", 0.0))
        
        # 1) Safety Reward: c_safe * U_{t+1}
        if cfg.use_safety:
            safety_reward = cfg.c_safe * upright
            total_reward += safety_reward
        
        # 2) Improvement Reward: w_t * [beta_u * max(0, Delta_U) + beta_p * max(0, Delta_P)]
        if cfg.use_improvement:
            # Only compute if we have previous values (skip first step)
            if self.prev_upright is not None:
                # Delta upright
                delta_U = max(0.0, upright - self.prev_upright)
                
                # Delta progress
                if self.prev_progress is not None:
                    delta_P = max(0.0, progress - self.prev_progress)
                else:
                    delta_P = 0.0
                
                improvement = (
                    cfg.beta_u * delta_U +
                    cfg.beta_p * delta_P
                )
                total_reward += weight_assist * improvement
            
            # Update previous values for next step
            self.prev_upright = upright
            self.prev_progress = progress
        
        # 3) Energy Penalty: -lambda_E * ||effective_intervention||^2
        if cfg.use_energy_penalty:
            energy_penalty = -cfg.lambda_E * np.linalg.norm(effective_intervention) ** 2
            total_reward += energy_penalty
        
        # 4) Usage Penalty (skill-weighted): -lambda_W(s) * w_t^p
        if cfg.use_usage_penalty:
            s = self.skill_level
            lambda_W_s = cfg.lambda_W0 * (1 + cfg.rho * s)
            usage_penalty = -lambda_W_s * (weight_assist ** cfg.p)
            total_reward += usage_penalty
        
        # 5) Responsibility & Goal
        if cfg.use_responsibility:
            is_fallen = bool(info.get("is_fallen", False))
            goal_reached = bool(info.get("goal_reached", False))
            
            # Responsibility-weighted fall penalty
            if is_fallen:
                fall_penalty = -cfg.c_fall * self.resp_ema
                total_reward += fall_penalty
            
            # Goal bonus
            if goal_reached:
                goal_bonus = cfg.b_goal
                total_reward += goal_bonus
        
        return float(total_reward)
    
    def update_coach(self, n_gradient_steps=10, learning_starts=1000):
        """Update coach agent and return comprehensive logs."""
        if self.coach_agent and self.coach_buffer and self.coach_buffer.size > learning_starts:
            logs = self.coach_agent.update_from_buffer(
                self.coach_buffer, 
                n_gradient_steps=n_gradient_steps
            )
            return logs
        return {}