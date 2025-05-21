"""
Gradient Bandit Agent implementation.

This module implements a Gradient Bandit agent that uses a softmax policy with gradient ascent.
The agent maintains preferences for each action and updates them based on received rewards.
"""

import numpy as np
from .base_agent import BaseAgent

class GradientBanditAgent(BaseAgent):
    """
    A Gradient Bandit agent that uses a softmax policy with gradient ascent.
    
    The agent maintains preferences H_t(a) for each action a and selects actions
    according to the softmax distribution. The preferences are updated using
    stochastic gradient ascent to maximize the expected reward.
    """
    
    def __init__(self, alpha=0.1, baseline=True):
        """
        Initialize the Gradient Bandit agent.
        
        Args:
            alpha (float): Step size parameter for the gradient update.
            baseline (bool): Whether to use the average reward as a baseline.
        """
        super().__init__("GradientBandit")
        self.alpha = alpha
        self.baseline = baseline
        self.preferences = None
        self.average_reward = 0.0
        self.step_count = 0
        
    def init_actions(self, n_actions):
        """
        Initialize the agent's internal state.
        
        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions)
        # Initialize preferences to zero
        self.preferences = np.zeros(n_actions)
        self.average_reward = 0.0
        self.step_count = 0
        
    def softmax(self, x):
        """Compute softmax values for each action."""
        # Subtract max for numerical stability
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    def get_action(self):
        """
        Select an action according to the softmax probability distribution.
        
        Returns:
            int: The index of the selected action.
        """
        if self.preferences is None:
            raise ValueError("Agent has not been initialized. Call init_actions() first.")
            
        # Compute action probabilities using softmax
        action_probs = self.softmax(self.preferences)
        
        # Sample an action according to the probabilities
        return np.random.choice(self.action_count, p=action_probs)
        
    def update(self, action, reward):
        """
        Update the agent's preferences based on the received reward.
        
        Args:
            action (int): The action that was taken.
            reward (float): The reward received.
        """
        if self.preferences is None:
            raise ValueError("Agent has not been initialized. Call init_actions() first.")
            
        # Update average reward if using baseline
        if self.baseline:
            self.step_count += 1
            self.average_reward += (reward - self.average_reward) / self.step_count
        
        # Compute action probabilities
        action_probs = self.softmax(self.preferences)
        
        # Compute the baseline term
        baseline_term = self.average_reward if self.baseline else 0.0
        
        # Update preferences using gradient ascent
        for a in range(self.action_count):
            if a == action:
                self.preferences[a] += self.alpha * (reward - baseline_term) * (1 - action_probs[a])
            else:
                self.preferences[a] -= self.alpha * (reward - baseline_term) * action_probs[a]
    
    @property
    def name(self):
        """Return the name of the agent with its parameters."""
        baseline_str = "with_baseline" if self.baseline else "no_baseline"
        return f"{self._name}(α={self.alpha}, {baseline_str})"
