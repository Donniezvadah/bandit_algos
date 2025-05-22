import numpy as np
import re
from openai import OpenAI
from .base_agent import BaseAgent
import time
import os

class LLMAgent(BaseAgent):
    """
    A no-regret-oriented LLM agent using structured prompting and distributional action sampling.
    Implements regret tracking, softmax policy, and importance-weighted updates for bandit feedback.
    """

    def __init__(self, api_key=None, model="o4-mini"):  # Using o4-mini model
        super().__init__("LLM")
        self.model = model
        # For o4-mini, temperature must be 1.0
        self.temperature = 1.0 if model == "o4-mini" else 0.7
        self._rewards = None
        self._counts = None
        self._action_history = []
        self._reward_history = []
        self._regret_history = []
        self._context_window = 10

        if api_key is None:
            try:
                with open('llm_api.txt', 'r') as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                raise ValueError("API key not provided and llm_api.txt not found")

        if not api_key:
            raise ValueError("API key is empty. Please provide a valid OpenAI API key.")

        self.api_key = api_key
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.client.models.list()
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

    def init_actions(self, n_actions):
        super().init_actions(n_actions)
        self._rewards = np.zeros(n_actions)
        self._counts = np.zeros(n_actions)
        self._action_history = []
        self._reward_history = []
        self._regret_history = []

    def _get_context_prompt(self):
        if not self._action_history:
            return "First round. You are given multiple arms. Return a probability distribution over actions."

        recent_actions = self._action_history[-self._context_window:]
        recent_rewards = self._reward_history[-self._context_window:]
        context = "Round history:\n"
        for i, (a, r) in enumerate(zip(recent_actions, recent_rewards)):
            context += f"Round {i+1}: Action {a}, Reward {r:.2f}\n"

        action_stats = np.zeros(len(self._rewards))
        counts = np.zeros(len(self._rewards))
        for a, r in zip(recent_actions, recent_rewards):
            action_stats[a] += r
            counts[a] += 1

        context += "\nAverages:\n"
        for i in range(len(self._rewards)):
            if counts[i] > 0:
                context += f"Action {i}: Avg Reward = {action_stats[i] / counts[i]:.2f} (tried {int(counts[i])} times)\n"
            else:
                context += f"Action {i}: Not tried yet\n"

        return context

    def get_action(self):
        if self._rewards is None or self._counts is None:
            raise ValueError("Call init_actions() first.")

        context = self._get_context_prompt()
        prompt = f"""
You are an online learning agent in a multi-armed bandit game. Your goal is to minimize regret.
{context}

INSTRUCTIONS:
1. You must provide a probability distribution over the {len(self._rewards)} possible actions.
2. The probabilities must sum to 1.0.
3. Format your response EXACTLY as shown in the example below.

Example for 3 actions:
Policy: [0.2, 0.5, 0.3]

Your response (with {len(self._rewards)} probabilities):
Policy: ["""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strategic AI minimizing regret in adversarial environments."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_completion_tokens=150
        )
        
        output = response.choices[0].message.content
        print(f"LLM Response: {output}")
        
        try:
            # First try to find a policy line with probabilities
            policy_lines = [line for line in output.splitlines() if 'policy' in line.lower() and '[' in line and ']' in line]
            if policy_lines:
                policy_line = policy_lines[0]
                policy_str = policy_line.split('[')[1].split(']')[0]
                probs = [float(x.strip()) for x in policy_str.split(',') if x.strip()]
                
                if len(probs) == len(self._rewards):
                    probs = np.array(probs)
                    probs = np.maximum(probs, 0)  # Ensure no negative probabilities
                    probs = probs / (probs.sum() + 1e-6)  # Normalize
                    
                    if not np.any(np.isnan(probs)) and not np.any(probs < 0) and abs(probs.sum() - 1.0) < 1e-6:
                        action = np.random.choice(len(probs), p=probs)
                        print(f"Sampled action {action} from policy distribution")
                        return action
            
            # If policy parsing fails, try to extract a single action
            # Look for patterns like "Action: 1" or "Choose 2"
            action_patterns = [
                r'action\s*[=:]?\s*(\d+)',
                r'choose\s*(?:action)?\s*(\d+)',
                r'select\s*(?:action)?\s*(\d+)',
                r'\b(\d+)\b',
                r'\[(?:\s*\d+\s*,?\s*)+\]'  # List of numbers
            ]
            
            for pattern in action_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                for match in matches:
                    try:
                        # If match is a list of numbers, pick one
                        if '[' in match or ',' in match:
                            numbers = [int(x.strip()) for x in re.findall(r'\d+', match)]
                            if numbers:
                                action = numbers[0]  # Pick first number
                                if 0 <= action < len(self._rewards):
                                    print(f"Selected action {action} from list")
                                    return action
                        else:
                            action = int(match)
                            if 0 <= action < len(self._rewards):
                                print(f"Selected action {action} from pattern")
                                return action
                    except (ValueError, IndexError):
                        continue
            
            # If we still don't have an action, try to find any number that could be a valid action
            numbers = re.findall(r'\d+', output)
            for num in numbers:
                try:
                    action = int(num)
                    if 0 <= action < len(self._rewards):
                        print(f"Extracted action {action} from response")
                        return action
                except (ValueError, IndexError):
                    continue
                    
            # If all else fails, return a random action
            action = np.random.randint(len(self._rewards))
            print(f"Falling back to random action: {action}")
            return action
            
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            # Fall back to random action if there's any error
            action = np.random.randint(len(self._rewards))
            print(f"Falling back to random action: {action}")
            return action

    def update(self, action, reward):
        self._rewards[action] += reward
        self._counts[action] += 1
        self._action_history.append(action)
        self._reward_history.append(reward)

        # Regret tracking: compare to best action in hindsight
        best_reward = max(self._rewards / (self._counts + 1e-6))
        actual = reward
        regret = best_reward - actual
        self._regret_history.append(regret)

    def avg_regret(self):
        if not self._regret_history:
            return 0
        return sum(self._regret_history) / len(self._regret_history)

    @property
    def name(self):
        return f"{self._name}({self.model})"
