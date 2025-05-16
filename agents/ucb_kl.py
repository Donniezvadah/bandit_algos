import numpy as np
from scipy.optimize import bisect
from .base_agent import BaseAgent

class KLUCBAgent(BaseAgent):
    """
    KL-UCB agent for Bernoulli bandit problem (sublinear optimality).
    Uses the KL-UCB index for each arm.
    """
    def __init__(self, n_arms: int, c: float = 3):
        super().__init__("KL-UCB")
        self.n_arms = n_arms
        self.c = c
        self.total_counts = 0

    def init_actions(self, n_actions):
        self.n_arms = n_actions
        self.total_counts = 0
        super().init_actions(n_actions)

    def kl_divergence(self, p, q):
        eps = 1e-15
        p = min(max(p, eps), 1 - eps)
        q = min(max(q, eps), 1 - eps)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def kl_ucb_index(self, p_hat, n, t, c):
        if n == 0:
            return 1.0
        upper_bound = 1.0
        lower_bound = p_hat
        rhs = (np.log(t) + c * np.log(np.log(max(t, 2)))) / n
        def func(q):
            return self.kl_divergence(p_hat, q) - rhs
        try:
            return bisect(func, lower_bound, upper_bound, xtol=1e-6)
        except Exception:
            return upper_bound

    def get_action(self):
        self.total_counts += 1
        t = self.total_counts
        indices = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            n = self.counts[arm]
            if n == 0:
                indices[arm] = 1.0
            else:
                p_hat = self.rewards[arm] / n
                indices[arm] = self.kl_ucb_index(p_hat, n, t, self.c)
        return int(np.argmax(indices))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.rewards[arm] += reward

    @property
    def name(self):
        return self._name 