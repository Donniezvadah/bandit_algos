# bandits/epsilon.py
from typing import Optional, Union
import numpy as np
from .base_agent import BaseAgent

class EpsilonGreedyAgent(BaseAgent):
    """
    An agent that explores with probability epsilon and exploits (chooses the best action so far)
    with probability 1-epsilon.
    """

    def __init__(self, epsilon: float = 0.01, environment_type: str = 'bernoulli'):
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
        self._successes: Optional[np.ndarray] = None
        self._failures: Optional[np.ndarray] = None
        self.rewards: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None
        self.environment_type = environment_type
        self.t: int = 0  # Time step counter

    def init_actions(self, n_actions: int) -> None:
        """
        Initializes the agent's internal state.

        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions)
        if self.environment_type == 'bernoulli':
            self._successes = np.zeros(n_actions)
            self._failures = np.zeros(n_actions)
        elif self.environment_type == 'gaussian':
            self.rewards = np.zeros(n_actions)
            self.counts = np.zeros(n_actions)
        else:
            raise ValueError(f"Unsupported environment type: {self.environment_type}")

    def get_action(self) -> int:
        """
        Chooses an action based on the epsilon-greedy strategy.

        Returns:
            int: The index of the chosen action.
        """
        self.t += 1  # Increment time step
        
        if self.environment_type == 'bernoulli':
            if self._successes is None or self._failures is None:
                raise ValueError("Agent has not been initialized. Call init_actions() first.")
            
            # Decaying epsilon: epsilon = 1/sqrt(t)
            current_epsilon = min(1.0, 1.0 / np.sqrt(self.t))
            
            if np.random.random() < current_epsilon:
                return np.random.randint(len(self._successes))
            
            # Calculate Q-values with proper handling of zero counts
            total_counts = self._successes + self._failures
            q_values = np.where(total_counts > 0,
                              self._successes / total_counts,
                              np.zeros_like(self._successes))
            return np.argmax(q_values)
            
        else:  # gaussian
            if self.rewards is None or self.counts is None:
                raise ValueError("Agent has not been initialized. Call init_actions() first.")
            
            if np.random.random() < self._epsilon:
                return np.random.randint(len(self.rewards))
            
            # Calculate Q-values with proper handling of zero counts
            q_values = np.where(self.counts > 0,
                              self.rewards / self.counts,
                              np.zeros_like(self.rewards))
            return np.argmax(q_values)

    def update(self, action: int, reward: Union[int, float]) -> None:
        """
        Updates the agent's internal state based on the action taken and reward received.

        Args:
            action (int): The action that was taken.
            reward (Union[int, float]): The reward received.
        """
        if self.environment_type == 'bernoulli':
            if reward == 1:
                self._successes[action] += 1
            else:
                self._failures[action] += 1
        else:  # gaussian
            if self.rewards is None or self.counts is None:
                raise ValueError("Agent has not been initialized. Call init_actions() first.")
            self.rewards[action] += reward
            self.counts[action] += 1

    @property
    def name(self) -> str:
        """Returns the name of the agent, including the epsilon value and environment type."""
        return f"{self._name}(epsilon={self._epsilon}, {self.environment_type})"
