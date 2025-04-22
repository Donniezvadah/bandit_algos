from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    def __init__(self, name):
        self._name = name
        self.action_count = None
        self.actions = None
        self.rewards = None
        self.counts = None
        
    def init_actions(self, n_actions):
        """Initialize the agent with n_actions possible actions."""
        self.action_count = n_actions
        self.actions = np.arange(n_actions)
        self.rewards = np.zeros(n_actions)
        self.counts = np.zeros(n_actions)
        
    @abstractmethod
    def get_action(self):
        """Select an action based on the agent's strategy."""
        pass
        
    def update(self, action, reward):
        """Update the agent's knowledge based on the action taken and reward received."""
        self.rewards[action] += reward
        self.counts[action] += 1
        
    @property
    def name(self):
        """Get the name of the agent."""
        return self._name 
    
    
    ## This is a comment