from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class BaseAgent(ABC):
    def __init__(self, name: str):
        self._name = name
        self.action_count: Optional[int] = None
        self.actions: Optional[np.ndarray] = None
        self.rewards: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None
        self.mean_rewards: Optional[np.ndarray] = None
        
    def init_actions(self, n_actions: int) -> None:
        """Initialize the agent with n_actions possible actions."""
        self.action_count = n_actions
        self.actions = np.arange(n_actions)
        self.rewards = np.zeros(n_actions)
        self.counts = np.zeros(n_actions)
        self.mean_rewards = np.zeros(n_actions)
        
    @abstractmethod
    def get_action(self) -> int:
        """Select an action based on the agent's strategy."""
        pass
        
    def update(self, action: int, reward: float) -> None:
        """Update the agent's knowledge based on the action taken and reward received."""
        if self.counts is None or self.rewards is None or self.mean_rewards is None:
            raise RuntimeError("Agent not initialized. Call init_actions() first.")
            
        self.rewards[action] += reward
        self.counts[action] += 1
        self.mean_rewards[action] = self.rewards[action] / self.counts[action]
        
    def reset(self) -> None:
        """Reset the agent's internal state."""
        if self.action_count is not None:
            self.rewards = np.zeros(self.action_count)
            self.counts = np.zeros(self.action_count)
            self.mean_rewards = np.zeros(self.action_count)
            self.actions = np.arange(self.action_count)
    
    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name     
    
    def choose_action(self) -> int:
        """Alias for get_action to maintain compatibility with existing code."""
        return self.get_action()