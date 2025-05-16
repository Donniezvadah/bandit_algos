import numpy as np
from .base_agent import BaseAgent

class GaussianEpsilonGreedyAgent(BaseAgent):
    """
    Epsilon-Greedy agent for Gaussian bandit environment.
    """
    def __init__(self, n_arms: int, epsilon: float = 0.1, c: float = 1.0):
        super().__init__("GaussianEpsilonGreedy")
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.c = c
        self.t = 0

    def init_actions(self, n_actions):
        self.n_arms = n_actions
        super().init_actions(n_actions)

    def get_action(self):
        self.t += 1
        current_epsilon = min(1.0, self.c / np.sqrt(self.t))
        means = self.rewards / (self.counts + 1e-6)
        if np.random.rand() < current_epsilon:
            return np.random.randint(self.n_arms)
        else:
            return int(np.argmax(means))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.rewards[arm] += reward

    @property
    def name(self):
        return self._name 