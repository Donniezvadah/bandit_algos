import numpy as np
from .base_agent import BaseAgent

class GaussianUCBAgent(BaseAgent):
    """
    UCB agent for Gaussian bandit environment using UCB1-Normal formula.
    """
    def __init__(self, n_arms: int, c: float = 2):
        super().__init__("GaussianUCB")
        self.n_arms = n_arms
        self.c = c
        self.total_counts = 0
        self.squared_sums = np.zeros(n_arms)  # For variance estimation

    def init_actions(self, n_actions):
        self.n_arms = n_actions
        self.total_counts = 0
        self.squared_sums = np.zeros(n_actions)
        super().init_actions(n_actions)

    def get_action(self):
        self.total_counts += 1
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        means = self.rewards / (self.counts + 1e-6)
        ucb_values = means + self.c * np.sqrt(2 * np.log(self.total_counts) / (self.counts + 1e-6))
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.squared_sums[arm] += reward ** 2

    @property
    def name(self):
        return self._name 