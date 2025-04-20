# bandits/epsilon.py
import numpy as np
from .base_agent import BaseAgent
import os

class EpsilonGreedyAgent(BaseAgent):
    """
    An agent that explores with probability epsilon and exploits (chooses the best action so far)
    with probability 1-epsilon.
    """

    def __init__(self, epsilon=0.01):
        """
        Initializes the EpsilonGreedyAgent.

        Args:
            epsilon (float): The probability of exploration (choosing a random action).
        """
        super().__init__("EpsilonGreedy")
        if not (0 <= epsilon <= 1):
            raise ValueError("Epsilon must be between 0 and 1")
        self._epsilon = epsilon
        self._successes = None  # Initialize successes and failures to None
        self._failures = None

    def init_actions(self, n_actions):
        """
        Initializes the agent's internal state (successes and failures counts for each action).

        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions)  # Call the parent's init_actions to initialize successes and failures
        self._successes = np.zeros(n_actions)  # Number of successes for each action
        self._failures = np.zeros(n_actions)   # Number of failures for each action

    def get_action(self):
        """
        Chooses an action based on the epsilon-greedy strategy.

        Returns:
            int: The index of the chosen action.
        """
        if self._successes is None or self._failures is None:
            raise ValueError("Agent has not been initialized. Call init_actions() first.")

        if np.random.random() < self._epsilon:
            # Explore: choose a random action
            return np.random.randint(len(self._successes))
        else:
            # Exploit: choose the action with the highest empirical success rate
            # Avoid division by zero by adding a small constant
            q_values = self._successes / (self._successes + self._failures + 1e-6)
            return np.argmax(q_values)

    def update(self, action, reward):
        """
        Updates the agent's internal state based on the action taken and reward received.

        Args:
            action (int): The action that was taken.
            reward (float): The reward received.
        """
        if reward == 1:
            self._successes[action] += 1
        else:
            self._failures[action] += 1

    @property
    def name(self):
        """Returns the name of the agent, including the epsilon value."""
        return f"{self._name}(epsilon={self._epsilon})"

agents = [
    EpsilonGreedyAgent(epsilon=0.1),
    UCBAgent(),
    ThompsonSamplingAgent(),
    LLMAgent(api_key=os.getenv('OPENAI_API_KEY'))  # Add LLM agent
]