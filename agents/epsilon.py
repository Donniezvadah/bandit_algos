# bandits/epsilon.py
import numpy as np
from .base_agent import BaseAgent

class EpsilonGreedyAgent(BaseAgent):
    """
    An agent that explores with probability epsilon and exploits (chooses the best action so far)
    with probability 1-epsilon.
    """

    def __init__(self, epsilon=0.01, environment_type='bernoulli'):
        """
        Initializes the EpsilonGreedyAgent.

        Args:
            epsilon (float): The probability of exploration (choosing a random action).
            environment_type (str): Type of environment ('bernoulli' or 'gaussian')
        """
        super().__init__("EpsilonGreedy")
        if not (0 <= epsilon <= 1):
            raise ValueError("Epsilon must be between 0 and 1")
        self._epsilon = epsilon
        self._successes = None
        self._failures = None
        self.environment_type = environment_type

    def init_actions(self, n_actions):
        """
        Initializes the agent's internal state.

        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions)
        if self.environment_type == 'bernoulli':
            self._successes = np.zeros(n_actions)
            self._failures = np.zeros(n_actions)
        else:  # gaussian
            self.rewards = np.zeros(n_actions)
            self.counts = np.zeros(n_actions)

    def get_action(self):
        """
        Chooses an action based on the epsilon-greedy strategy.

        Returns:
            int: The index of the chosen action.
        """
        if self.environment_type == 'bernoulli':
            if self._successes is None or self._failures is None:
                raise ValueError("Agent has not been initialized. Call init_actions() first.")
            if np.random.random() < self._epsilon:
                return np.random.randint(len(self._successes))
            else:
                q_values = self._successes / (self._successes + self._failures + 1e-6)
                return np.argmax(q_values)
        else:  # gaussian
            if self.rewards is None or self.counts is None:
                raise ValueError("Agent has not been initialized. Call init_actions() first.")
            if np.random.random() < self._epsilon:
                return np.random.randint(len(self.rewards))
            else:
                q_values = self.rewards / (self.counts + 1e-6)
                return np.argmax(q_values)

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
            super().update(action, reward)

    @property
    def name(self):
        """Returns the name of the agent, including the epsilon value and environment type."""
        return f"{self._name}(epsilon={self._epsilon}, {self.environment_type})"
