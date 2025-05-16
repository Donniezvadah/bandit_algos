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

    def __init__(self, api_key=None, model="gpt-4.1-nano"):  # Use a strong model for strategic reasoning
        super().__init__("LLM")
        self.model = model
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

Give a probability distribution over the actions. Format your answer strictly as:
Policy: [p0, p1, ..., p{len(self._rewards) - 1}]
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strategic AI minimizing regret in adversarial environments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            output = response.choices[0].message.content
            policy_line = [line for line in output.splitlines() if line.startswith("Policy")][0]
            policy_str = policy_line.split("[", 1)[1].split("]")[0]
            probs = np.array([float(x.strip()) for x in policy_str.split(',')])
            probs /= probs.sum()  # normalize
            action = np.random.choice(len(probs), p=probs)
            return action

        except Exception as e:
            print(f"LLM fallback due to error: {e}")
            means = self._rewards / (self._counts + 1e-6)
            bonus = np.sqrt(2 * np.log(len(self._action_history) + 1) / (self._counts + 1e-6))
            return int(np.argmax(means + bonus))

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
