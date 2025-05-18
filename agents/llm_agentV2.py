import numpy as np
from openai import OpenAI
from .base_agent import BaseAgent
import time
import os

class LLMAgent(BaseAgent):
    """
    A no-regret-oriented LLM agent using structured prompting and distributional action sampling.
    Implements regret tracking, softmax policy, and importance-weighted updates for bandit feedback.
    """

    def __init__(self, api_key=None, model="gpt-4"):  # Using gpt-4 for better reasoning
        super().__init__("LLM")
        self.model = model
        self._rewards = None
        self._counts = None
        self._action_history = []
        self._reward_history = []
        self._regret_history = []
        self._context_window = 20  # Increased context window for better pattern recognition

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
        total_rounds = len(self._action_history)
        context = []
        
        # Basic information
        context.append(f"You are playing a multi-armed bandit game with {len(self._rewards)} arms.")
        context.append(f"Current round: {total_rounds + 1}\n")
        
        # If first round
        if not self._action_history:
            context.append("This is the first round. Explore the arms to learn their rewards.")
            return "\n".join(context)
            
        # Calculate statistics for each arm
        arm_stats = []
        for arm in range(len(self._rewards)):
            mask = np.array(self._action_history) == arm
            if np.any(mask):
                rewards = np.array(self._reward_history)[mask]
                arm_stats.append({
                    'arm': arm,
                    'count': np.sum(mask),
                    'total_reward': np.sum(rewards),
                    'avg_reward': np.mean(rewards),
                    'last_pulled': len(self._action_history) - np.where(mask)[0][-1] - 1 if np.any(mask) else float('inf'),
                    'ucb': np.mean(rewards) + np.sqrt(2 * np.log(total_rounds) / np.sum(mask)) if np.sum(mask) > 0 else float('inf')
                })
            else:
                arm_stats.append({
                    'arm': arm,
                    'count': 0,
                    'total_reward': 0,
                    'avg_reward': 0,
                    'last_pulled': float('inf'),
                    'ucb': float('inf')
                })
        
        # Sort by UCB score (Upper Confidence Bound)
        arm_stats.sort(key=lambda x: x['ucb'], reverse=True)
        
        # Build context
        context.append("Arm Statistics (sorted by potential):")
        context.append("Arm |  Count  |  Total Reward  |  Avg Reward  |  Last Pulled  |  UCB Score")
        context.append("-" * 80)
        
        for stat in arm_stats:
            context.append(
                f"{stat['arm']:>3} | "
                f"{stat['count']:>7} | "
                f"{stat['total_reward']:>13.2f} | "
                f"{stat['avg_reward']:>11.2f} | "
                f"{stat['last_pulled']:>12} | "
                f"{stat['ucb']:>9.2f}"
            )
        
        # Add recent history
        context.append("\nRecent History (last 5 actions):")
        recent_history = list(zip(
            self._action_history[-5:],
            [f"{r:.2f}" for r in self._reward_history[-5:]]
        ))
        for i, (a, r) in enumerate(recent_history, 1):
            context.append(f"Round {total_rounds - len(recent_history) + i}: Arm {a} → Reward {r}")
        
        return "\n".join(context)

    def get_action(self):
        if self._rewards is None or self._counts is None:
            raise ValueError("Call init_actions() first.")

        context = self._get_context_prompt()
        prompt = f"""
You are an advanced multi-armed bandit learning agent. Your goal is to maximize cumulative reward by balancing exploration and exploitation.

{context}

Guidelines:
1. Favor arms with higher UCB scores (balance of average reward and exploration bonus)
2. Consider how recently each arm was pulled
3. Maintain some exploration probability for less-tried arms
4. Focus more on promising arms (high average reward with sufficient trials)

Return a probability distribution over the arms. The sum must be exactly 1.0.
Format your response exactly as:
Policy: [p0, p1, ..., p{len(self._rewards) - 1}]

Example: Policy: [0.1, 0.6, 0.2, 0.1]"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strategic AI minimizing regret in adversarial environments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent decisions
            max_tokens=150
        )
        
        output = response.choices[0].message.content
        policy_line = [line for line in output.splitlines() if line.startswith("Policy")][0]
        policy_str = policy_line.split("[", 1)[1].split("]")[0]
        probs = np.array([float(x.strip()) for x in policy_str.split(',')])
        probs = np.maximum(probs, 0)  # Ensure no negative probabilities
        probs = probs / (probs.sum() + 1e-6)  # Normalize with small epsilon for numerical stability
        
        if np.any(np.isnan(probs)) or np.any(probs < 0) or abs(probs.sum() - 1.0) > 1e-6:
            raise ValueError(f"Invalid probability distribution generated: {probs}")
            
        action = np.random.choice(len(probs), p=probs)
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
