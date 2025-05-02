# bandits/thompson.py
import numpy as np
from .base_agent import BaseAgent
from abc import ABCMeta, abstractmethod, abstractproperty

class ThompsonSamplingAgent(BaseAgent):
    """
    An agent that uses Thompson Sampling to select actions. Thompson Sampling maintains a Beta
    distribution for each action, representing the uncertainty about its reward probability, and
    samples from these distributions to choose the next action.
    """

    def __init__(self, environment_type='bernoulli'):
        """
        Initialize the ThompsonSamplingAgent.
        
        Args:
            environment_type (str): Type of environment ('bernoulli' or 'gaussian')
        """
        super().__init__("ThompsonSampling")
        self._successes = None
        self._failures = None
        self._rewards = None
        self._counts = None
        self._sum_squared_rewards = None
        self.environment_type = environment_type

    def init_actions(self, n_actions):
        """
        Initializes the agent's internal state.

        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions)
        if self.environment_type == 'bernoulli':
            self._successes = np.ones(n_actions)  # Initialize with 1 to avoid zero probabilities
            self._failures = np.ones(n_actions)
        else:  # gaussian
            self._rewards = np.zeros(n_actions)
            self._counts = np.zeros(n_actions)
            self._sum_squared_rewards = np.zeros(n_actions)

    def get_action(self):
        """
        Chooses an action based on Thompson Sampling.

        Returns:
            int: The index of the chosen action.
        """
        if self.environment_type == 'bernoulli':
            if self._successes is None or self._failures is None:
                raise ValueError("Agent has not been initialized. Call init_actions() first.")
            # Sample from Beta distributions for each action
            samples = np.random.beta(self._successes, self._failures)
        else:  # gaussian
            if self._rewards is None or self._counts is None:
                raise ValueError("Agent has not been initialized. Call init_actions() first.")
            # Sample from Normal distributions for each action
            means = self._rewards / (self._counts + 1e-6)
            variances = (self._sum_squared_rewards / (self._counts + 1e-6)) - means**2
            variances = np.maximum(variances, 0)  # Ensure non-negative
            stds = np.sqrt(variances / (self._counts + 1e-6))
            samples = np.random.normal(means, stds)
            
        return np.argmax(samples)

    def update(self, action, reward):
        """
        Updates the agent's internal state based on the action taken and reward received.

        Args:
            action (int): The action that was taken.
            reward (float): The reward received.
        """
        if self.environment_type == 'bernoulli':
            if reward == 1:
                self._successes[action] += 1
            else:
                self._failures[action] += 1
        else:  # gaussian
            self._rewards[action] += reward
            self._sum_squared_rewards[action] += reward * reward
            self._counts[action] += 1

    @property
    def name(self):
        """Returns the name of the agent."""
        return f"{self._name}({self.environment_type})"
    
    # Thompson Sampling is a Bayesian approach to the exploration-exploitation trade-off.
    # It maintains a Beta distribution for each action, representing the uncertainty about its reward probability, and
    # samples from these distributions to choose the next action.
    # The Beta distribution is a conjugate prior for the Bernoulli likelihood, which makes it a natural choice for modeling the probability of success for a given action.
    # The Beta distribution has two parameters, alpha and beta, which represent the number of successes and failures, respectively.
    # The mean of the Beta distribution is alpha / (alpha + beta), which is the expected reward probability for the action.
    # The variance of the Beta distribution is alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1)), which measures the uncertainty about the reward probability.
    # Thompson Sampling works by sampling from the Beta distributions for each action and choosing the action with the highest sample value.
    # This approach balances exploration and exploitation by leveraging the Beta distribution's properties to balance the exploration-exploitation trade-off.
    # Comments