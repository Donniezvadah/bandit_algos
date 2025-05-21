import numpy as np
import random
import time
import os
import re
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from functools import lru_cache

from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from .base_agent import BaseAgent

class LLMAgent(BaseAgent):
    """
    An agent that uses OpenAI API to make decisions in the bandit environment.
    This agent maintains a history of actions and rewards to provide context to the LLM.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4.1-nano", 
                 temperature: float = 0.3, max_retries: int = 3, 
                 timeout: int = 30, cache_dir: str = '.llm_cache'):
        """
        Initialize the LLM agent with enhanced error handling and caching.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from llm_api.txt.
            model: The model to use. Default is "gpt-4.1-nano".
            temperature: Controls randomness in the response (0-1).
            max_retries: Maximum number of retries for API calls.
            timeout: Timeout in seconds for API calls.
            cache_dir: Directory to store API response caches.
        """
        super().__init__("LLM")
        self.model = model
        self.temperature = max(0, min(1, temperature))  # Clamp between 0 and 1
        self.max_retries = max(1, max_retries)
        self.timeout = max(5, timeout)  # Minimum 5 second timeout
        self._cache_dir = cache_dir
        
        # Initialize state
        self._rewards = None
        self._counts = None
        self._action_history = []
        self._reward_history = []
        self._context_window = 20  # Increased context window
        self._last_api_call = 0
        self._min_call_interval = 0.1  # 100ms between API calls to avoid rate limiting
        
        # Set up API client
        if api_key is None:
            try:
                with open('llm_api.txt', 'r') as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                raise ValueError("API key not provided and llm_api.txt not found")
        
        if not api_key:
            raise ValueError("API key is empty. Please provide a valid OpenAI API key.")
        
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        
        # Create cache directory if it doesn't exist
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # Test the API connection
        self._test_connection()

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

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a cache key for the given prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load a response from the cache if it exists and is recent."""
        cache_file = os.path.join(self._cache_dir, f"{cache_key}.json")
        if not os.path.exists(cache_file):
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                # Check if cache is still valid (1 day)
                if time.time() - cache_data['timestamp'] < 86400:  # 24 hours
                    return cache_data['response']
        except (json.JSONDecodeError, KeyError, IOError):
            pass
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save a response to the cache."""
        cache_file = os.path.join(self._cache_dir, f"{cache_key}.json")
        cache_data = {
            'timestamp': time.time(),
            'response': response,
            'model': self.model,
            'temperature': self.temperature
        }
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except IOError:
            pass  # Don't fail if cache save fails
    
    def _throttle_api_calls(self):
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self._min_call_interval:
            time.sleep(self._min_call_interval - elapsed)
        self._last_api_call = time.time()
    
    def _test_connection(self):
        """Test the API connection with retries."""
        for attempt in range(self.max_retries):
            try:
                self._throttle_api_calls()
                self.client.models.list()
                return  # Success
            except (APIError, APITimeoutError, RateLimitError) as e:
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to connect to OpenAI API after {self.max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _get_context_prompt(self) -> str:
        """Generate a detailed context prompt for the LLM based on recent history."""
        if not self._action_history:
            return "This is the first action. You should explore different actions to learn their reward probabilities."
        
        # Get recent history
        recent_actions = self._action_history[-self._context_window:]
        recent_rewards = self._reward_history[-self._context_window:]
        
        # Calculate statistics
        action_counts = np.bincount(recent_actions, minlength=len(self._rewards))
        action_rewards = np.zeros(len(self._rewards))
        action_variances = np.zeros(len(self._rewards))
        
        # Track rewards for each action to calculate variance
        action_reward_lists = [[] for _ in range(len(self._rewards))]
        for a, r in zip(recent_actions, recent_rewards):
            action_rewards[a] += r
            action_reward_lists[a].append(r)
        
        # Calculate mean and variance for each action
        for a in range(len(self._rewards)):
            if action_counts[a] > 0:
                action_rewards[a] /= action_counts[a]
                if len(action_reward_lists[a]) > 1:
                    action_variances[a] = np.var(action_reward_lists[a])
        
        # Generate context
        context = "You are playing a multi-armed bandit game with the following actions available:\n"
        context += f"Available actions: {list(range(len(self._rewards)))}\n\n"
        
        context += "Recent history (last 20 steps):\n"
        context += "Step\tAction\tReward\n"
        context += "----------------------\n"
        for i, (a, r) in enumerate(zip(recent_actions, recent_rewards)):
            context += f"{i+1:4d}\t{a:3d}\t{r:.2f}\n"
        
        context += "\nAction statistics:\n"
        context += "Action\tCount\tAvg Reward\tVariance\n"
        context += "----------------------------------\n"
        for a in range(len(self._rewards)):
            if action_counts[a] > 0:
                context += f"{a:3d}\t{action_counts[a]:3d}\t{action_rewards[a]:.3f}\t\t{action_variances[a]:.3f}\n"
            else:
                context += f"{a:3d}\t  0\t  N/A\t\t  N/A\n"
        
        # Add exploration guidance
        context += "\nGuidance:\n"
        context += "- Your goal is to maximize the total reward over time.\n"
        context += "- Balance exploration (trying actions with high uncertainty) with exploitation (choosing actions with high average rewards).\n"
        context += "- Consider both the mean reward and the variance when making decisions.\n"
        
        return context

    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API with retries and error handling.
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            The response text from the LLM.
            
        Raises:
            RuntimeError: If all retry attempts fail.
        """
        cache_key = self._get_cache_key(prompt)
        
        # Try to load from cache first
        cached_response = self._load_from_cache(cache_key)
        if cached_response is not None:
            print("Using cached response")
            return cached_response
        
        # Prepare messages
        messages = [
            {"role": "system", "content": """You are an expert at playing multi-armed bandit games. 
            Your goal is to maximize the cumulative reward by balancing exploration and exploitation."""},
            {"role": "user", "content": prompt}
        ]
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self._throttle_api_calls()
                print(f"Calling OpenAI API (attempt {attempt + 1}/{self.max_retries}) with model: {self.model}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=100,
                    timeout=self.timeout
                )
                
                response_text = response.choices[0].message.content.strip()
                print(f"API Response: {response_text[:100]}...")  # Log first 100 chars
                
                # Cache the successful response
                self._save_to_cache(cache_key, response_text)
                return response_text
                
            except (APIError, APITimeoutError, RateLimitError) as e:
                last_error = e
                wait_time = (2 ** attempt) + (random.random() * 0.5)  # Add jitter
                print(f"API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                last_error = e
                print(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                break
        
        # If we get here, all retries failed
        raise RuntimeError(f"Failed to get response from LLM after {self.max_retries} attempts: {str(last_error)}")
    
    def _parse_action_from_response(self, response_text: str) -> int:
        """
        Parse the action from the LLM response.
        
        Args:
            response_text: The raw response text from the LLM.
            
        Returns:
            The parsed action index.
            
        Raises:
            ValueError: If no valid action can be parsed.
        """
        # Try to find action in various formats
        patterns = [
            r'action\s*(\d+)',  # "action 1"
            r'choose\s*action\s*(\d+)',  # "choose action 1"
            r'\b(\d+)\b',  # Just a number
            r'"action"\s*:\s*(\d+)',  # JSON-like: "action": 1
            r'"action"\s*:\s*"(\d+)"',  # JSON-like: "action": "1"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                try:
                    action = int(matches[0])
                    if 0 <= action < len(self._rewards):
                        print(f"Parsed action {action} from response")
                        return action
                except (ValueError, IndexError):
                    continue
        
        raise ValueError(f"Could not parse action from response: {response_text}")
    
    def get_action(self) -> int:
        """
        Choose an action using the LLM.
        
        Returns:
            int: The index of the chosen action.
            
        Raises:
            ValueError: If the agent is not initialized or if the LLM response is invalid.
            RuntimeError: If there are issues with the LLM API calls.
        """
        if self._rewards is None or self._counts is None:
            raise ValueError("Agent has not been initialized. Call init_actions() first.")
        
        # If any action hasn't been tried yet, try it first
        if np.any(self._counts == 0):
            action = int(np.argmin(self._counts))
            print(f"Exploring untried action: {action}")
            return action
        
        # Generate prompt with detailed context
        context = self._get_context_prompt()
        prompt = f"""You are playing a multi-armed bandit game with {len(self._rewards)} actions.
Your goal is to maximize the cumulative reward over time.

{context}

Based on the above information, which action should you choose next?

Please respond with ONLY the action number (0-{len(self._rewards)-1}) and a VERY brief explanation (1-2 sentences).
For example:
1  # This action has the highest estimated reward based on current data

Your response must be in the format: 'N # explanation' where N is the action number (0-{len(self._rewards)-1})

Your response:"""

        # Call the LLM API with retries and caching
        response_text = self._call_llm_api(prompt)
        
        # Parse the action from the response
        action = self._parse_action_from_response(response_text)
        print(f"LLM chose action {action}")
        
        # Validate the action is within bounds
        if not (0 <= action < len(self._rewards)):
            raise ValueError(f"LLM returned invalid action {action}, must be between 0 and {len(self._rewards)-1}")
            
        return action
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API with retries and error handling.
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            The response text from the LLM.
            
        Raises:
            RuntimeError: If all retry attempts fail.
        """
        cache_key = self._get_cache_key(prompt)
        
        # Try to load from cache first
        cached_response = self._load_from_cache(cache_key)
        if cached_response is not None:
            print("Using cached response")
            return cached_response
        
        # Prepare messages
        messages = [
            {"role": "system", "content": """You are an expert at playing multi-armed bandit games. 
            Your goal is to maximize the cumulative reward by balancing exploration and exploitation."""},
            {"role": "user", "content": prompt}
        ]
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self._throttle_api_calls()
                print(f"Calling OpenAI API (attempt {attempt + 1}/{self.max_retries}) with model: {self.model}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=100,
                    timeout=self.timeout
                )
                
                response_text = response.choices[0].message.content.strip()
                print(f"API Response: {response_text[:100]}...")  # Log first 100 chars
                
                # Cache the successful response
                self._save_to_cache(cache_key, response_text)
                return response_text
                
            except (APIError, APITimeoutError, RateLimitError) as e:
                last_error = e
                wait_time = (2 ** attempt) + (random.random() * 0.5)  # Add jitter
                print(f"API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                last_error = e
                print(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                break
        
        # If we get here, all retries failed
        raise RuntimeError(f"Failed to get response from LLM after {self.max_retries} attempts: {str(last_error)}")

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
    
    