import numpy as np
import openai
from .base_agent import BaseAgent
import time

class LLMAgent(BaseAgent):
    """
    An agent that uses ChatGPT API to make decisions in the bandit environment.
    This agent maintains a history of actions and rewards to provide context to the LLM.
    """

    def __init__(self, api_key=None, model="gpt-4"):
        """
        Initialize the LLM agent.
        
        Args:
            api_key (str): OpenAI API key. If None, will try to get from environment.
            model (str): The OpenAI model to use.
        """
        super().__init__("LLM")
        self.api_key = api_key
        self.model = model
        self._rewards = None
        self._counts = None
        self._action_history = []
        self._reward_history = []
        self._context_window = 10  # Number of recent actions to include in context
        
        # Initialize OpenAI client
        if self.api_key:
            openai.api_key = self.api_key
        else:
            raise ValueError("OpenAI API key is required for LLM agent")

    def init_actions(self, n_actions):
        """
        Initialize the agent's internal state.
        
        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions)
        self._rewards = np.zeros(n_actions)
        self._counts = np.zeros(n_actions)
        self._action_history = []
        self._reward_history = []

    def _get_context_prompt(self):
        """Generate the context prompt for the LLM based on recent history."""
        if not self._action_history:
            return "This is the first action. Choose an action to explore."
        
        # Get recent history
        recent_actions = self._action_history[-self._context_window:]
        recent_rewards = self._reward_history[-self._context_window:]
        
        # Calculate statistics
        action_counts = np.bincount(recent_actions, minlength=len(self._rewards))
        action_rewards = np.zeros(len(self._rewards))
        for a, r in zip(recent_actions, recent_rewards):
            action_rewards[a] += r
        
        # Generate context
        context = "Recent history:\n"
        for i, (a, r) in enumerate(zip(recent_actions, recent_rewards)):
            context += f"Step {i+1}: Action {a}, Reward {r:.2f}\n"
        
        context += "\nAction statistics:\n"
        for a in range(len(self._rewards)):
            if action_counts[a] > 0:
                avg_reward = action_rewards[a] / action_counts[a]
                context += f"Action {a}: Pulled {action_counts[a]} times, Average reward: {avg_reward:.2f}\n"
            else:
                context += f"Action {a}: Not tried yet\n"
        
        return context

    def get_action(self):
        """
        Choose an action using ChatGPT API.
        
        Returns:
            int: The index of the chosen action.
        """
        if self._rewards is None or self._counts is None:
            raise ValueError("Agent has not been initialized. Call init_actions() first.")

        # If any action hasn't been tried yet, try it
        if np.any(self._counts == 0):
            return np.argmin(self._counts)

        # Generate prompt for ChatGPT
        context = self._get_context_prompt()
        prompt = f"""You are playing a multi-armed bandit game. Your goal is to maximize cumulative reward.
{context}

Based on this information, which action should you choose next? 
Return only the action number (0-{len(self._rewards)-1}) and a brief explanation."""

        try:
            # Call ChatGPT API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at playing multi-armed bandit games."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            # Parse response to get action
            response_text = response.choices[0].message.content
            action = int(response_text.split()[0])  # Get first number in response
            
            # Validate action
            if not (0 <= action < len(self._rewards)):
                action = np.argmin(self._counts)  # Fallback to least tried action
            
            return action
            
        except Exception as e:
            print(f"Error calling ChatGPT API: {e}")
            # Fallback to UCB-like strategy
            means = self._rewards / (self._counts + 1e-6)
            exploration_bonus = np.sqrt(2 * np.log(len(self._action_history) + 1) / (self._counts + 1e-6))
            return np.argmax(means + exploration_bonus)

    def update(self, action, reward):
        """
        Update the agent's internal state based on the action taken and reward received.
        
        Args:
            action (int): The action that was taken.
            reward (float): The reward received.
        """
        self._rewards[action] += reward
        self._counts[action] += 1
        self._action_history.append(action)
        self._reward_history.append(reward)

    @property
    def name(self):
        """Returns the name of the agent."""
        return f"{self._name}({self.model})" 