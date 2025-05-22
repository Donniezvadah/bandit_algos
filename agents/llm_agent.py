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

    def __init__(self, api_key: str = None, model: str = "o4-mini", 
                 temperature: float = 1.0, max_retries: int = 3, 
                 timeout: int = 30, cache_dir: str = '.llm_cache'):
        """
        Initialize the LLM agent with enhanced error handling and caching.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from llm_api.txt.
            model: The model to use. Default is "o4-mini".
            temperature: Controls randomness in the response (0-1).
            max_retries: Maximum number of retries for API calls.
            timeout: Timeout in seconds for API calls.
            cache_dir: Directory to store API response caches (not used in this version).
        """
        super().__init__("LLM")
        self.model = model
        # For o4-mini, temperature must be 1.0
        if model == "o4-mini":
            self.temperature = 1.0
        else:
            self.temperature = max(0, min(1, temperature))  # Clamp between 0 and 1 for other models
        self.max_retries = max(1, max_retries)
        self.timeout = max(5, timeout)  # Minimum 5 second timeout
        
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
        
        # Test the API connection
        self._test_connection()

    def reset(self):
        """Reset the agent's internal state."""
        if hasattr(self, 'action_count') and self.action_count is not None:
            self._rewards = np.zeros(self.action_count)
            self._counts = np.zeros(self.action_count)
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
        self._action_history = []
        self._reward_history = []
    
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
        """Generate a detailed context prompt with complete history of loss vectors."""
        if not self._action_history:
            return "This is the first action. You should explore different actions to learn their reward probabilities."
        
        n_actions = len(self._rewards)
        
        # Calculate loss vectors for each step
        # Loss = 1 - reward (since rewards are in [0,1])
        loss_vectors = []
        for a, r in zip(self._action_history, self._reward_history):
            loss_vec = [1.0] * n_actions  # Initialize with maximum loss
            loss_vec[a] = 1.0 - r  # Actual loss for the taken action
            loss_vectors.append(loss_vec)
        
        # Generate context with complete history
        context = """You are playing a multi-armed bandit game with sequential decisions.
At each step, you choose an action and observe a loss (1 - reward).
Your goal is to minimize the cumulative loss over time.

Available actions: {}

Complete history of loss vectors (step: [loss_action_0, loss_action_1, ...]):
""".format(list(range(n_actions)))
        
        # Add all loss vectors
        for t, loss_vec in enumerate(loss_vectors, 1):
            formatted_losses = [f"{l:.3f}" for l in loss_vec]
            context += f"Step {t}: [{', '.join(formatted_losses)}]\n"
        
        # Add statistics about each action
        action_counts = np.bincount(self._action_history, minlength=n_actions)
        action_avg_loss = np.ones(n_actions)  # Initialize with max loss
        
        for a in range(n_actions):
            if action_counts[a] > 0:
                # Calculate average loss for this action
                action_rewards = [r for act, r in zip(self._action_history, self._reward_history) if act == a]
                action_avg_loss[a] = 1.0 - (sum(action_rewards) / len(action_rewards))
        
        context += "\nAction statistics (count, average loss):\n"
        for a in range(n_actions):
            if action_counts[a] > 0:
                context += f"Action {a}: chosen {action_counts[a]} times, avg loss = {action_avg_loss[a]:.3f}\n"
            else:
                context += f"Action {a}: never chosen (unknown loss)\n"
        
        # Add guidance
        context += """
Guidance:
- Your goal is to minimize the cumulative loss over time.
- You should balance exploration (trying actions with uncertain losses) with exploitation (choosing actions with known low losses).
- Consider both the average loss and the number of times each action has been tried when making decisions.
- Remember that the loss for untried actions is unknown (shown as 1.0).
"""
        
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
        # Prepare messages - no caching to ensure fresh responses each time
        messages = [
            {"role": "system", "content": """You are an expert at playing multi-armed bandit games. 
            Your goal is to maximize the cumulative reward by balancing exploration and exploitation.
            Consider the entire history of actions and rewards when making your decision."""},
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
                    max_completion_tokens=100,  # Updated for o4-mini compatibility
                    timeout=self.timeout
                )
                
                response_text = response.choices[0].message.content.strip()
                print(f"API Response: {response_text[:100]}...")  # Log first 100 chars
                
                # Don't cache the response to ensure the LLM learns from each run
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
        if not response_text or not response_text.strip():
            raise ValueError("Empty response from LLM")
            
        # Clean and normalize the response text
        response_text = response_text.strip().lower()
        print(f"Parsing response: {response_text}")
        
        # Try to find action in various formats
        patterns = [
            r'action\s*(\d+)',  # "action 1"
            r'select\s*action\s*(\d+)',  # "select action 1"
            r'choose\s*action\s*(\d+)',  # "choose action 1"
            r'\b(\d+)\b',  # Just a number
            r'"action"\s*:\s*(\d+)',  # JSON-like: "action": 1
            r'"action"\s*:\s*"(\d+)"',  # JSON-like: "action": "1"
            r'action\s*=\s*(\d+)',  # "action=1" or "action = 1"
            r'choice\s*[=:]?\s*(\d+)',  # "choice: 1" or "choice=1"
            r'\[\s*\d+\s*,\s*(\d+)\s*\]',  # [0, 1, 2] - pick the first number
            r'\b(?:option|action|choice)\s*[#:]?\s*(\d+)\b',  # "option 1", "action: 2", "choice #3"
            r'\b(?:i choose|i select|i pick|selecting|choosing)\s*(?:action\s*)?(\d+)\b',  # "I choose action 1"
        ]
        
        # First, try to find a number that's a valid action index
        for pattern in patterns:
            try:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                for match in matches:
                    try:
                        action = int(match)
                        if 0 <= action < len(self._rewards):
                            print(f"Parsed action {action} from pattern: {pattern}")
                            return action
                    except (ValueError, IndexError):
                        continue
            except Exception as e:
                print(f"Error with pattern {pattern}: {str(e)}")
                continue
        
        # If no pattern matched, try to extract any number that could be a valid action
        numbers = re.findall(r'\d+', response_text)
        for num in numbers:
            try:
                action = int(num)
                if 0 <= action < len(self._rewards):
                    print(f"Extracted action {action} from number in response")
                    return action
            except (ValueError, IndexError):
                continue
        
        # Try to find a policy distribution
        try:
            policy_match = re.search(r'\[([^\]]+)\]', response_text)
            if policy_match:
                probs = []
                for x in policy_match.group(1).split(','):
                    x = x.strip()
                    if x.replace('.', '').isdigit():
                        probs.append(float(x))
                
                if probs and len(probs) == len(self._rewards):
                    probs = np.array(probs)
                    probs = np.maximum(probs, 0)
                    if np.sum(probs) > 0:
                        probs = probs / np.sum(probs)
                        action = np.random.choice(len(probs), p=probs)
                        print(f"Sampled action {action} from policy distribution")
                        return action
        except Exception as e:
            print(f"Error parsing policy distribution: {str(e)}")
        
        # If we still don't have an action, try to find the first valid number in the response
        for word in response_text.split():
            try:
                action = int(word)
                if 0 <= action < len(self._rewards):
                    print(f"Found valid action {action} in response")
                    return action
            except (ValueError, IndexError):
                continue
        
        # If all else fails, look for any number that could be a valid action index
        for i in range(len(self._rewards)):
            if str(i) in response_text:
                print(f"Found action {i} mentioned in response")
                return i
        
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
            raise ValueError("Agent not initialized. Call init_actions() first.")
        
        # Generate prompt with detailed context
        context = self._get_context_prompt()
        prompt = f"""{context}
        
        Based on the above information, which action would you choose? 
        
        INSTRUCTIONS:
        1. You MUST respond with a SINGLE INTEGER between 0 and {len(self._rewards)-1}, inclusive.
        2. The integer should be the index of the action you want to take.
        3. Example valid responses: '0', '1', '2', etc.
        
        Your response (just the number, e.g., '2'): """
        
        try:
            # Get response from LLM
            response_text = self._call_llm_api(prompt)
            
            # Parse the action from the response
            action = self._parse_action_from_response(response_text)
            
            # Update history
            self._action_history.append(action)
            
            return action
            
        except Exception as e:
            # Fallback to random action if there's an error
            print(f"Error getting action from LLM: {str(e)}. Falling back to random action.")
            action = np.random.randint(len(self._rewards))
            self._action_history.append(action)
            return action
    
    def update(self, action, reward):
        """
        Update the agent's internal state based on the action taken and reward received.
        
        Args:
            action (int): The action that was taken.
            reward (float): The reward received.
        """
        if self._rewards is None or self._counts is None:
            return
            
        # Update counts and rewards
        self._counts[action] += 1
        n = self._counts[action]
        self._rewards[action] = ((n - 1) * self._rewards[action] + reward) / n
        
        # Update history
        if len(self._reward_history) < len(self._action_history):
            self._reward_history.append(reward)
        else:
            self._reward_history[-1] = reward
    
    @property
    def name(self):
        """Returns the name of the agent."""
        return f"LLM ({self.model})"
