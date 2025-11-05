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
        self.rollout_learning_progress = []  # Student's learning progress (LP_t)
        self.rollout_assistance_discounts = []  # Assistance discount factors (ψ(w_t))
        
        # Responsibility tracking (EMA of weight_assist)
        self.resp_ema = 0.0
        
        # Pending transition for correct MDP (s' includes a_{t+1})
        self._pending_transition = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self.episode_return = 0.0
        self.episode_coach_rewards = []
        self.rollout_weight_assists = []
        self.rollout_effective_intervention_norms = []
        self.rollout_learning_progress = []
        self.rollout_assistance_discounts = []
        self.resp_ema = 0.0
        self._pending_transition = None
        return obs, info

    def step(self, student_action):
        # 0) Pass through if no coach
        if self.coach_agent is None:
            return self.env.step(student_action)
        
        # 1) Commit previous transition if exists (now we have a_{t+1}^S)
        if self.coach_buffer is not None and self._pending_transition is not None:
            # Complete previous transition with current student_action as a_{t+1}^S
            next_coach_obs = np.concatenate([
                self._last_obs,
                [self.skill_level],
                student_action
            ])
            
            self.coach_buffer.add(
                self._pending_transition["coach_obs"],
                self._pending_transition["coach_action"],
                self._pending_transition["coach_reward"],
                next_coach_obs,
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
        resp_ema_alpha = 0.1
        self.resp_ema = resp_ema_alpha * weight_assist + (1 - resp_ema_alpha) * self.resp_ema
        
        # 5) Compute interventions
        intended_intervention = weight_assist * self.kappa * coach_action
        exec_action = np.clip(student_action + intended_intervention, -1.0, 1.0)
        effective_intervention = exec_action - student_action
        
        # 6) Run environment step
        next_obs, reward, terminated, truncated, info = self.env.step(exec_action)
        
        # 7) Compute coach reward using baseline design
        coach_reward, coach_metrics = self._compute_coach_reward(
            info=info,
            coach_action=coach_action,
            weight_assist=weight_assist
        )
        self.episode_coach_rewards.append(coach_reward)
        
        # 8) Track statistics
        self.rollout_weight_assists.append(weight_assist)
        self.rollout_effective_intervention_norms.append(np.linalg.norm(effective_intervention))
        self.rollout_learning_progress.append(coach_metrics["LP"])
        self.rollout_assistance_discounts.append(coach_metrics["psi"])
        
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
                    dummy_next_action
                ])
                
                self.coach_buffer.add(
                    coach_obs,
                    coach_action,
                    coach_reward,
                    next_coach_obs_terminal,
                    True,
                    weight_assist
                )
                self._pending_transition = None
        
        # 10) Update skill estimator at episode end
        if (terminated or truncated) and self.estimator:
            if "episode" in info and "r" in info["episode"]:
                self.skill_level = self.estimator.update(info["episode"]["r"])
        
        # 11) Extend info with step-level metrics
        info["weight_assist"] = weight_assist
        info["effective_intervention_norm"] = float(np.linalg.norm(effective_intervention))
        info["skill_level"] = self.skill_level
        info["resp_ema"] = self.resp_ema
        info["student_learning_progress"] = coach_metrics["LP"]
        info["assistance_discount"] = coach_metrics["psi"]
        
        # Add episode-level rollout statistics
        if terminated or truncated:
            info["rollout/coach_reward_mean"] = float(np.mean(self.episode_coach_rewards))
            info["rollout/coach_reward_sum"] = float(np.sum(self.episode_coach_rewards))
            info["rollout/weight_assist_mean"] = float(np.mean(self.rollout_weight_assists))
            info["rollout/weight_assist_std"] = float(np.std(self.rollout_weight_assists))
            info["rollout/effective_intervention_norm_mean"] = float(np.mean(self.rollout_effective_intervention_norms))
            info["rollout/skill_level"] = float(self.skill_level)
            info["rollout/student_learning_progress_mean"] = float(np.mean(self.rollout_learning_progress))
            info["rollout/student_learning_progress_std"] = float(np.std(self.rollout_learning_progress))
            info["rollout/assistance_discount_mean"] = float(np.mean(self.rollout_assistance_discounts))
        
        self._last_obs = next_obs
        self.episode_return += reward
        
        return next_obs, reward, terminated, truncated, info
    
    def _compute_assistance_discount(self, weight_assist: float) -> float:
        """
        Compute ψ(w_t) = (1 - min{w_t/ρ, 1})^2
        
        This discounts the learning progress based on how much the coach assisted.
        - w_t = 0 → ψ = 1.0 (full credit to student)
        - w_t = ρ → ψ = 0.0 (no credit, coach did everything)
        """
        rho = self.reward_cfg.rho
        ratio = min(weight_assist / rho, 1.0)
        psi = (1.0 - ratio) ** 2
        return float(psi)
    
    def _compute_coach_reward(self, info: dict, coach_action: np.ndarray, 
                              weight_assist: float) -> tuple[float, dict]:
        """
        Baseline Coach Reward Design:
        
        LP_t = ψ(w_t) · r_t^shaped
        r_coach = LP_t - β*w_t - λ_E*||a^C||^2
        
        where:
        - r_t^shaped: shaped reward from environment (excludes terminal signals)
        - ψ(w_t): assistance discount function
        - β: usage penalty coefficient
        - λ_E: energy penalty coefficient
        """
        cfg = self.reward_cfg
        
        # 1. Get shaped reward from environment info
        r_shaped = float(info.get("reward_shaped", 0.0))
        
        # 2. Compute assistance discount
        psi = self._compute_assistance_discount(weight_assist)
        
        # 3. Student's Learning Progress (discounted by assistance)
        LP = psi * r_shaped
        
        # 4. Penalties
        usage_penalty = cfg.beta * weight_assist
        action_norm_sq = float(np.sum(coach_action ** 2))
        energy_penalty = cfg.lambda_E * action_norm_sq
        
        # 5. Total coach reward
        coach_reward = LP - usage_penalty - energy_penalty
        
        metrics = {
            "LP": LP,
            "psi": psi,
            "r_shaped": r_shaped,
            "usage_penalty": usage_penalty,
            "energy_penalty": energy_penalty,
        }
        
        return float(coach_reward), metrics
    
    def update_coach(self, n_gradient_steps=10, learning_starts=1000):
        """Update coach agent and return comprehensive logs."""
        if self.coach_agent and self.coach_buffer and self.coach_buffer.size > learning_starts:
            logs = self.coach_agent.update_from_buffer(
                self.coach_buffer, 
                n_gradient_steps=n_gradient_steps
            )
            return logs
        return {}