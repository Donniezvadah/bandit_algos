import numpy as np
from environments.bernoulli_bandit import BernoulliBandit
from agents.ucb import UCBAgent  # Using UCB agent for testing
import matplotlib.pyplot as plt

def test_llm_agent():
    # Set up a simple bandit environment with 3 arms
    n_actions = 3
    probs = [0.1, 0.5, 0.9]  # Easy case: clear best action
    env = BernoulliBandit(n_actions=n_actions, probs=probs)
    
    # Initialize the UCB agent for testing
    agent = UCBAgent(
        alpha=2.0,  # Exploration parameter
        confidence=1.96  # 95% confidence interval
    )
    
    # Run a short simulation
    n_steps = 20
    n_trials = 3
    
    print(f"Testing LLM agent on {n_actions}-armed bandit with {n_trials} trials of {n_steps} steps")
    print(f"Arm probabilities: {probs}")
    
    # Track rewards and actions
    rewards = np.zeros(n_steps)
    action_counts = np.zeros(n_actions)
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        env.reset()
        agent.init_actions(n_actions)
        
        for step in range(n_steps):
            # Get action and reward
            action = agent.get_action()
            reward = env.pull(action)
            
            # Update agent
            agent.update(action, reward)
            
            # Track statistics
            rewards[step] += reward
            action_counts[action] += 1
            
            print(f"Step {step + 1}: Action {action}, Reward {reward}")
    
    # Calculate average rewards
    avg_rewards = rewards / n_trials
    action_proportions = action_counts / (n_trials * n_steps)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot average reward over time
    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards, 'b-', label='Average Reward')
    plt.axhline(y=max(probs), color='r', linestyle='--', label='Optimal Reward')
    plt.xlabel('Step')
    plt.ylabel('Average Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    
    # Plot action selection distribution
    plt.subplot(1, 2, 2)
    plt.bar(range(n_actions), action_proportions)
    plt.xticks(range(n_actions))
    plt.xlabel('Action')
    plt.ylabel('Proportion of Selections')
    plt.title('Action Selection Distribution')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('test_llm_agent_results.png')
    print("\nTest complete. Results saved to 'test_llm_agent_results.png'")
    print(f"Final action distribution: {action_proportions}")
    print(f"Optimal action is {np.argmax(probs)} with probability {max(probs)}")

if __name__ == "__main__":
    test_llm_agent()
