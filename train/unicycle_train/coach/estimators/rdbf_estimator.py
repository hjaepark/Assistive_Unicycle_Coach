# train/unicycle_train/coach/estimators/rdbf_estimator.py

import torch
import numpy as np
from typing import Tuple, List, Optional

class RDBFEstimator:
    """
    Recursive Discrete Bayes Filter (RDBF) for Skill Estimation.
    
    Based on: Skill Estimator Design.pdf
    - Estimates belief state b_t over 3 discrete skill levels: {Novice, Intermediate, Expert}.
    - Uses a pre-trained Expert Policy as the reference distribution.
    """

    def __init__(
        self, 
        action_dim: int, 
        action_min: torch.Tensor, 
        action_max: torch.Tensor, 
        beta: float = 0.5, 
        transition_matrix: Optional[torch.Tensor] = None,
        device: str = 'cpu'
    ):
        """
        Initialize the RDBF Estimator.

        Args:
            action_dim (int): Dimension of the action space (e.g., 2 for Wheel & Pitch).
            action_min (Tensor): Minimum bounds of action space (e.g., [-1, -1]).
            action_max (Tensor): Maximum bounds of action space (e.g., [1, 1]).
            beta (float): Mixing coefficient for Intermediate level [0, 1].
                          P(I) = beta * P(E) + (1-beta) * P(N).
            transition_matrix (Tensor, optional): State transition probabilities T(phi_t | phi_{t-1}).
                                                  Defaults to a stable identity-like matrix.
            device (str): Computation device ('cpu' or 'cuda').
        """
        self.device = device
        self.levels = ['Novice', 'Intermediate', 'Expert']
        self.num_levels = len(self.levels)
        
        # 1. Belief Initialization: b_0 = [1/3, 1/3, 1/3]
        #
        self.belief = torch.ones(self.num_levels, device=self.device) / self.num_levels
        
        # 2. Novice Model Parameters (Uniform Distribution)
        # P(a|s, N) = 1 / (a_max - a_min) for each dimension (Assuming independence or volume)
        #
        self.action_range = (action_max - action_min).to(self.device)
        self.novice_prob_density = 1.0 / torch.prod(self.action_range)
        
        # 3. Intermediate Model Parameter
        #
        self.beta = beta
        
        # 4. Transition Matrix (Prediction Model)
        # T[i][j] = P(phi_t = i | phi_{t-1} = j)
        #
        if transition_matrix is not None:
            self.transition_matrix = transition_matrix.to(self.device)
        else:
            # Default: High inertia to prevent flickering (Stable Estimate)
            # Diagonal = 0.99, Small chance to transition to adjacent levels
            self.transition_matrix = torch.tensor([
                [0.99, 0.01, 0.00],  # Novice -> Novice, Inter
                [0.01, 0.98, 0.01],  # Inter -> Novice, Inter, Expert
                [0.00, 0.01, 0.99]   # Expert -> Inter, Expert
            ], device=self.device).T # Transpose for Matrix Multiplication (Cols -> Rows)

    def reset(self):
        """Resets the belief to the initial uniform distribution."""
        self.belief = torch.ones(self.num_levels, device=self.device) / self.num_levels

    def update(self, student_action: torch.Tensor, expert_dist: torch.distributions.Distribution) -> Tuple[torch.Tensor, int]:
        """
        Performs the Recursive Bayesian Update step.
        
        Corresponds to the "Final Equation":
        b_t = eta * P(a_t | s_t, phi) * sum(T * b_{t-1})
       

        Args:
            student_action (Tensor): The action taken by the student (a_t).
            expert_dist (Distribution): The Gaussian distribution from the Expert Policy pi*(.|s_t).
                                       

        Returns:
            belief (Tensor): Updated probability vector [P(N), P(I), P(E)].
            skill_level (int): Index of the most likely skill (0, 1, or 2).
        """
        
        # --- Step 1: Calculate Observation Likelihoods P(a_t | s_t, phi) ---
        
        # 1. Expert Likelihood P(a|s, E): Derived from pre-trained policy (Gaussian)
        #
        # Note: log_prob returns log-likelihood, exp() converts it to probability density.
        expert_log_prob = expert_dist.log_prob(student_action)
        
        # If action is multi-dimensional, sum log_probs (assuming independence) before exp
        if expert_log_prob.dim() > 0: 
             expert_log_prob = expert_log_prob.sum()
             
        prob_expert = torch.exp(expert_log_prob)

        # 2. Novice Likelihood P(a|s, N): Uniform Distribution
        #
        prob_novice = self.novice_prob_density
        
        # 3. Intermediate Likelihood P(a|s, I): Mixture Model
        #
        prob_inter = (self.beta * prob_expert) + ((1 - self.beta) * prob_novice)
        
        # Construct Likelihood Vector: [P(a|N), P(a|I), P(a|E)]
        likelihood_vector = torch.tensor([prob_novice, prob_inter, prob_expert], device=self.device)
        
        # --- Step 2: Prediction from Previous Belief (Time Update) ---
        
        # Apply Transition Matrix: sum_{j} T(phi^i | phi^j) * b_{t-1}(phi^j)
        #
        # self.transition_matrix is already transposed: T_matrix @ belief_vector
        predicted_belief = torch.matmul(self.transition_matrix, self.belief)
        
        # --- Step 3: Measurement Update (Correction) ---
        
        # Unnormalized Belief = Likelihood * Predicted_Belief
        unnormalized_belief = likelihood_vector * predicted_belief
        
        # --- Step 4: Normalization (eta) ---
        # Ensure sum(b_t) = 1
        self.belief = unnormalized_belief / (unnormalized_belief.sum() + 1e-8)
        
        # Get the most likely skill index
        current_skill_idx = torch.argmax(self.belief).item()
        
        return self.belief, current_skill_idx

    def get_skill_distribution(self) -> dict:
        """Returns the current belief as a dictionary for logging."""
        return {
            "prob_novice": self.belief[0].item(),
            "prob_intermediate": self.belief[1].item(),
            "prob_expert": self.belief[2].item()
        }