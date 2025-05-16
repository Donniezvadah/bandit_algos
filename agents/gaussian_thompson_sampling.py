import numpy as np
from .base_agent import BaseAgent

class GaussianThompsonSamplingAgent(BaseAgent):
    """
    Thompson Sampling agent for Gaussian bandit environment using Normal-Inverse-Gamma conjugate prior.
    This implementation maintains posterior parameters for each arm and samples from the posterior predictive distribution.
    """
    def __init__(self, n_arms: int):
        super().__init__("GaussianThompsonSampling")
        self.n_arms = n_arms
        self.reset()

    def init_actions(self, n_actions):
        self.n_arms = n_actions
        self.reset()
        super().init_actions(n_actions)

    def reset(self):
        self.mu0 = 0.0
        self.lambda0 = 1.0
        self.alpha0 = 1.0
        self.beta0 = 1.0
        self.n = np.zeros(self.n_arms)
        self.sum_x = np.zeros(self.n_arms)
        self.sum_x2 = np.zeros(self.n_arms)

    def get_action(self):
        samples = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            n = self.n[arm]
            if n == 0:
                mu_n = self.mu0
                lambda_n = self.lambda0
                alpha_n = self.alpha0
                beta_n = self.beta0
            else:
                sample_mean = self.sum_x[arm] / n
                lambda_n = self.lambda0 + n
                mu_n = (self.lambda0 * self.mu0 + n * sample_mean) / lambda_n
                alpha_n = self.alpha0 + n / 2
                sum_sq = self.sum_x2[arm]
                beta_n = self.beta0 + 0.5 * (sum_sq - n * sample_mean ** 2) + \
                    (self.lambda0 * n * (sample_mean - self.mu0) ** 2) / (2 * lambda_n)
            tau2 = 1 / np.random.gamma(alpha_n, 1 / beta_n)
            samples[arm] = np.random.normal(mu_n, np.sqrt(tau2 / lambda_n))
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        self.n[arm] += 1
        self.sum_x[arm] += reward
        self.sum_x2[arm] += reward ** 2

    @property
    def name(self):
        return self._name 