import numpy as np
from .base_environment import BaseEnvironment

class GaussianBandit(BaseEnvironment):
    def __init__(self, means=None, stds=None, n_actions=10, reward_scale=1.0):
        """
        Initialize a Gaussian bandit environment optimized for Thompson sampling.
        
        Args:
            means (list or np.array): List of means for each arm. If None, means are generated with good separation.
            stds (list or np.array): List of standard deviations for each arm. If None, stds are generated with reasonable values.
            n_actions (int): Number of actions if means and stds are not provided.
            reward_scale (float): Scale factor for rewards to ensure reasonable magnitudes.
        """
        super().__init__()
        
        # Set number of actions
        self.action_count = n_actions
        
        # Generate or validate means
        if means is not None:
            means = np.array(means, dtype=float)
            if len(means) != n_actions:
                raise ValueError(f"Number of means ({len(means)}) must match number of actions ({n_actions})")
            self.means = means
        else:
            # Generate means with good separation for optimal regret
            # Use a logarithmic spacing to ensure good separation between arms
            self.means = np.logspace(0, 1, n_actions, base=2)
            
        # Generate or validate standard deviations
        if stds is not None:
            stds = np.array(stds, dtype=float)
            if len(stds) != n_actions:
                raise ValueError(f"Number of standard deviations ({len(stds)}) must match number of actions ({n_actions})")
            if not np.all(stds > 0):
                raise ValueError("All standard deviations must be positive")
            self.stds = stds
        else:
            # Use varying standard deviations to test Thompson sampling's ability to handle uncertainty
            # Higher variance for arms with higher means to increase the challenge
            self.stds = np.linspace(0.5, 2.0, n_actions)
            
        # Scale the rewards to ensure reasonable magnitudes
        self.means *= reward_scale
        self.stds *= reward_scale
        
        # Store initial state
        self._initial_means = np.copy(self.means)
        self._initial_stds = np.copy(self.stds)
        
        # Compute optimal action and value
        self._optimal_action = np.argmax(self.means)
        self._optimal_mean = self.means[self._optimal_action]
        
        # Store additional statistics for analysis
        self._total_pulls = np.zeros(n_actions)
        self._total_rewards = np.zeros(n_actions)
        self._sum_squared_rewards = np.zeros(n_actions)
        
    def pull(self, action):
        """
        Pull the specified arm and return the reward.
        
        Args:
            action (int): The index of the action to take.
            
        Returns:
            float: The reward from the selected action.
            
        Raises:
            ValueError: If the action is out of bounds.
        """
        if not (0 <= action < self.action_count):
            raise ValueError(f"Action {action} is out of bounds. Must be between 0 and {self.action_count - 1}")
        
        # Generate reward from normal distribution
        reward = np.random.normal(self.means[action], self.stds[action])
        
        # Update statistics
        self._total_pulls[action] += 1
        self._total_rewards[action] += reward
        self._sum_squared_rewards[action] += reward**2
        
        return float(reward)
        
    def optimal_reward(self):
        """
        Return the optimal reward (maximum mean).
        
        Returns:
            float: The maximum mean among all actions.
        """
        return float(self._optimal_mean)
        
    def step(self):
        """
        Step the environment. In this implementation, it does nothing as it's a stationary environment.
        """
        pass
        
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.means = np.copy(self._initial_means)
        self.stds = np.copy(self._initial_stds)
        self._optimal_action = np.argmax(self.means)
        self._optimal_mean = self.means[self._optimal_action]
        self._total_pulls = np.zeros(self.action_count)
        self._total_rewards = np.zeros(self.action_count)
        self._sum_squared_rewards = np.zeros(self.action_count)
        
    def get_statistics(self):
        """
        Return statistics about the environment's performance.
        
        Returns:
            dict: Dictionary containing various statistics about the environment.
        """
        return {
            'total_pulls': self._total_pulls,
            'total_rewards': self._total_rewards,
            'sum_squared_rewards': self._sum_squared_rewards,
            'empirical_means': self._total_rewards / (self._total_pulls + 1e-10),
            'empirical_stds': np.sqrt(
                (self._sum_squared_rewards / (self._total_pulls + 1e-10)) - 
                (self._total_rewards / (self._total_pulls + 1e-10))**2
            )
        } 