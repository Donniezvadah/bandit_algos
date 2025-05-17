# bandits/ucb.py
import numpy as np
from .base_agent import BaseAgent

class UCBAgent(BaseAgent):
    """
    An agent that uses the Upper Confidence Bound (UCB1) algorithm to select actions.
    UCB1 balances exploration and exploitation by adding an exploration bonus to the estimated
    reward of each action.
    """

    def __init__(self):
        """Initializes the UCBAgent."""
        super().__init__("UCB")
        self._rewards = None
        self._counts = None
        self._total_steps = 0

    def init_actions(self, n_actions):
        """
        Initializes the agent's internal state.

        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions)
        self._rewards = np.zeros(n_actions)
        self._counts = np.zeros(n_actions)
        self._total_steps = 0

    def get_action(self):
        """
        Chooses an action based on the UCB1 algorithm.

        Returns:
            int: The index of the chosen action.
        """
        if self._rewards is None or self._counts is None:
            raise ValueError("Agent has not been initialized. Call init_actions() first.")

        # If any action hasn't been tried yet, try it
        if np.any(self._counts == 0):
            return np.argmin(self._counts)

        means = self._rewards / (self._counts + 1e-6)
        exploration_bonus = 3 * np.sqrt(np.log(self._total_steps + 1) / (self._counts + 1e-6))
        ucb_values = means + exploration_bonus
        return np.argmax(ucb_values)

    def update(self, action, reward):
        """
        Updates the agent's internal state based on the action taken and reward received.

        Args:
            action (int): The action that was taken.
            reward (float): The reward received.
        """
        self._rewards[action] += reward
        self._counts[action] += 1
        self._total_steps += 1

    @property
    def name(self):
        """Returns the name of the agent."""
        return self._name
    