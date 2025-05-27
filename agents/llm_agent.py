import numpy as np
import random
import time
import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from .base_agent import BaseAgent

from typing import Optional, List, Tuple
import numpy as np

class LLMAgent(BaseAgent):
    """
    An agent that uses OpenAI API to make decisions in the bandit environment.
    This agent maintains a history of actions and rewards to provide context to the LLM.
    The agent uses its own reasoning capabilities to balance exploration and exploitation
    based on the history of actions and their outcomes.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-nano", 
                 temperature: float = 0.0, max_retries: int = 3, 
                 timeout: int = 30, max_history_length: int = 100):
        """
        Initialize the LLM agent.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from llm_api.txt.
            model: The model to use. Default is "gpt-4.1-nano".
            temperature: Controls randomness in the response (0-1).
            max_retries: Maximum number of retries for API calls.
            timeout: Timeout in seconds for API calls.
            max_history_length: Maximum length of action/reward history to keep.
        """
        super().__init__("LLM")
        self.model = model
        self.temperature = max(0, min(1, temperature))  # Clamp between 0 and 1
        self.max_retries = max(1, max_retries)
        self.timeout = max(5, timeout)  # Minimum 5 second timeout
        self.max_history_length = max(1, max_history_length)
        
        # Initialize state
        self._rewards: Optional[np.ndarray] = None
        self._counts: Optional[np.ndarray] = None
        self._successes: Optional[np.ndarray] = None
        self._failures: Optional[np.ndarray] = None
        self._action_history: List[int] = []
        self._reward_history: List[float] = []
        self._last_api_call = 0
        self._min_call_interval = 0.1  # 100ms between API calls to avoid rate limiting
        
        # Set up API client
        if api_key is None:
            try:
                with open('llm_api.txt', 'r') as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                print("Warning: API key not provided and llm_api.txt not found.")
                raise ValueError("API key is required for LLMAgent")
        
        self.api_key = api_key
        self.client: Optional[OpenAI] = None
        
        try:
            self.client = OpenAI(api_key=self.api_key)
            # Test the API connection
            self._test_connection()
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise

    def reset(self):
        """Reset the agent's internal state."""
        if hasattr(self, 'action_count') and self.action_count is not None:
            self._rewards = np.zeros(self.action_count)
            self._counts = np.zeros(self.action_count)
            self._successes = np.zeros(self.action_count)
            self._failures = np.zeros(self.action_count)
        self._action_history = []
        self._reward_history = []
    
    def init_actions(self, n_actions):
        """
        Initialize the agent's internal state.
        
        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions)
        self._rewards = np.zeros(n_actions)
        self._counts = np.zeros(n_actions)
        self._successes = np.zeros(n_actions)
        self._failures = np.zeros(n_actions)
        self._action_history = []
        self._reward_history = []
    
    def _throttle_api_calls(self) -> None:
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self._min_call_interval:
            time.sleep(self._min_call_interval - elapsed)
        self._last_api_call = time.time()
        
    def _handle_api_error(self, error: Exception, attempt: int) -> None:
        """Handle API errors with appropriate backoff strategy."""
        if isinstance(error, RateLimitError):
            print(f"Rate limit exceeded. Waiting for {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
        elif isinstance(error, (APIError, APITimeoutError)):
            print(f"API error occurred: {str(error)}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
        else:
            print(f"Unexpected error: {str(error)}")
            raise error
    
    def _test_connection(self) -> None:
        """Test the API connection with retries."""
        for attempt in range(self.max_retries):
            try:
                self._throttle_api_calls()
                self.client.models.list()
                return  # Success
            except Exception as e:
                self._handle_api_error(e, attempt)
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to connect to OpenAI API after {self.max_retries} attempts: {str(e)}")
    
    def _get_context_prompt(self) -> str:
        """
        Generate a detailed context prompt with action history and statistics.
        
        Returns:
            Formatted context string for the LLM
        """
        if not self._action_history:
            return "This is the first action. You should explore different actions to learn their reward probabilities."
        
        n_actions = len(self._rewards)
        total_steps = len(self._action_history)
        
        # Calculate statistics for each action
        action_stats = []
        for a in range(n_actions):
            if self._counts[a] > 0:
                rewards = [r for act, r in zip(self._action_history, self._reward_history) if act == a]
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards, ddof=1) if len(rewards) > 1 else 0
                success_rate = self._successes[a] / self._counts[a] if self._counts[a] > 0 else 0
                action_stats.append({
                    'action': a,
                    'count': int(self._counts[a]),
                    'successes': int(self._successes[a]),
                    'failures': int(self._failures[a]),
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'success_rate': success_rate
                })
            else:
                action_stats.append({
                    'action': a,
                    'count': 0,
                    'successes': 0,
                    'failures': 0,
                    'mean_reward': None,
                    'std_reward': None,
                    'success_rate': None
                })
        
        # Generate context
        context = f"""You are an advanced decision-making AI playing a multi-armed bandit game with {n_actions} actions.
Your goal is to minimize cumulative regret by choosing actions that maximize expected reward.

Current statistics (step {total_steps}):
"""
        
        # Add action statistics
        context += "\nAction Statistics:"
        context += "\n| Action | Count | Successes | Failures | Mean Reward | Std Dev | Success Rate |"
        context += "\n|--------|-------|-----------|----------|-------------|---------|--------------|"
        
        for stats in action_stats:
            if stats['count'] > 0:
                context += f"\n| {stats['action']:6d} | {stats['count']:5d} | {stats['successes']:9d} | {stats['failures']:8d} | {stats['mean_reward']:.4f}    | {stats['std_reward']:.4f}  | {stats['success_rate']:.4f}    |"
            else:
                context += f"\n| {stats['action']:6d} |     0 |         0 |        0 |      ?.????    |   ?.????  |      ?.????    |"
        
        # Add recent history
        context += "\n\nRecent History (last 5 steps):\n"
        context += "| Step | Action | Reward |\n"
        context += "|------|--------|--------|\n"
        
        # Calculate actual step numbers
        total_steps = len(self._action_history)
        for i in range(max(0, len(self._action_history)-5), len(self._action_history)):
            actual_step = total_steps - (len(self._action_history) - i)
            context += f"| {actual_step:4d} | {self._action_history[i]:6d} | {self._reward_history[i]:.4f} |\n"
        
        # Add guidance
        context += """

Guidance:
1. Your goal is to minimize cumulative regret by maximizing expected reward.
2. Consider both the mean reward and uncertainty (std dev) of each action.
3. Balance exploration (trying actions with high uncertainty) with exploitation (choosing actions with high mean rewards).
4. For untried actions, consider exploring them to gather more information.
5. Look for patterns in the recent history that might indicate changes in reward distributions.
6. Use the success rate and failure counts to assess the reliability of each action.

Please respond with just the action number (0-{}) that you think will minimize regret.""".format(n_actions-1)
        
        return context

    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API with retries and error handling.
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            The LLM's response as a string.
            
        Raises:
            ValueError: If the API key is invalid or API connection fails.
            RuntimeError: If the API response is invalid or unexpected.
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized")
            
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps with decision making in multi-armed bandit problems. Respond with just the action number."},
            {"role": "user", "content": prompt}
        ]
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self._throttle_api_calls()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=50,
                    timeout=self.timeout
                )
                
                # Extract the response text
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content.strip()
                else:
                    raise ValueError("Empty response from LLM")
                    
            except (APIError, APITimeoutError, RateLimitError) as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Unexpected error after {self.max_retries} attempts: {str(e)}")
                time.sleep(1)  # Shorter delay for non-rate-limit errors
        
        raise RuntimeError(f"Failed to get response from LLM: {str(last_error)}")
    
    def _parse_llm_response(self, response: str) -> int:
        """
        Parse the LLM's response to extract the chosen action.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            The chosen action (0 to n_actions-1) or None if parsing fails
        """
        if not response:
            return None
            
        # Clean the response
        response = response.strip()
        
        # Look for patterns like "Action X" or "Choose X"
        patterns = [
            r'action\s*(\d+)',
            r'choose\s*(\d+)',
            r'select\s*(\d+)',
            r'\b(\d+)\b'  # Any number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    action = int(match.group(1))
                    if 0 <= action < self.action_count:
                        return action
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def choose_action(self) -> int:
        """
        Choose an action using the LLM's reasoning capabilities.
        The agent uses its history and statistics to make informed decisions.
        """
        if self._rewards is None or self.action_count == 0:
            return np.random.randint(0, self.action_count)
        
        try:
            # Prepare context for the LLM
            context = self._get_context_prompt()
            
            # Call the LLM
            response = self._call_llm_api(context)
            
            # Parse the response to get the chosen action
            action = self._parse_llm_response(response)
            
            # Ensure the action is valid
            if action is not None and 0 <= action < self.action_count:
                return action
                
        except Exception as e:
            print(f"Error in LLM decision making: {e}")
            # If all else fails, choose randomly
            return np.random.randint(0, self.action_count)
    
    def update(self, action: int, reward: float) -> None:
        """Update the agent's knowledge based on the action taken and reward received."""
        if self._counts is None or self._rewards is None or self._successes is None or self._failures is None:
            raise RuntimeError("Agent not initialized. Call init_actions() first.")
            
        self._rewards[action] += reward
        self._counts[action] += 1
        
        # Update successes and failures
        if reward > 0:
            self._successes[action] += 1
        else:
            self._failures[action] += 1
        
        # Use incremental update for numerical stability
        if self._counts[action] == 1:
            self._rewards[action] = reward
        else:
            # Update the mean incrementally
            self._rewards[action] += (reward - self._rewards[action]) / self._counts[action]
        
        # Update history
        self._action_history.append(action)
        self._reward_history.append(reward)
        
        # Trim history if it gets too large
        if len(self._action_history) > 1000:  # Keep last 1000 steps
            self._action_history = self._action_history[-1000:]
            self._reward_history = self._reward_history[-1000:]
    
    @property
    def name(self):
        """Return the name of the agent."""
        return f"LLM ({self.model}, temp={self.temperature})"

    def get_action(self) -> int:
        """
        Required method from the abstract base class.
        This is an alias for choose_action to maintain compatibility.
        
        Returns:
            int: The index of the chosen action.
        """
        return self.choose_action()
