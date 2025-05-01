# bandits/main.py
import numpy as np
import matplotlib.pyplot as plt
from environments.bernoulli_bandit import BernoulliBandit
from agents.llm_agent import LLMAgent
from utils.confidence import compute_confidence_interval
from plots.plot_utils import plot_regret_with_confidence
import configparser
import xml.etree.ElementTree as ET
import os
from environments.gaussian_bandit import GaussianBandit

def load_config():
    """Load configuration from XML file."""
    tree = ET.parse('config/config.xml')
    root = tree.getroot()
    
    config = {
        'paths': {
            'plots_dir': root.find('paths/plots_dir').text,
            'agents_dir': root.find('paths/agents_dir').text,
            'environments_dir': root.find('paths/environments_dir').text
        },
        'simulation': {
            'n_steps': int(root.find('simulation/n_steps').text),
            'n_trials': int(root.find('simulation/n_trials').text),
            'confidence_levels': [float(level.text) for level in root.findall('simulation/confidence_levels/level')]
        },
        'seeds': {
            'numpy': int(root.find('seeds/numpy').text),
            'random': int(root.find('seeds/random').text)
        },
        'environments': {
            'n_actions': int(root.find('environments/n_actions').text),
            'bernoulli': {
                'probs': [float(prob.text) for prob in root.findall('environments/bernoulli/probs/prob')]
            }
        }
    }
    return config

def run_simulation(env, agent, n_steps, n_trials, confidence_levels):
    """Run the bandit simulation with a single agent."""
    regrets = np.zeros((n_trials, n_steps))
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Reset environment and agent
        env.reset()
        agent.init_actions(env.action_count)
        
        # Run simulation
        for step in range(n_steps):
            if (step + 1) % 10 == 0:  # Print more frequently
                print(f"Step {step + 1}/{n_steps}", end='\r')
            
            # Get action from agent
            action = agent.get_action()
            
            # Get reward from environment
            reward = env.pull(action)
            
            # Print action and reward
            print(f"Action: {action}, Reward: {reward}")
            
            # Update agent
            agent.update(action, reward)
            
            # Calculate regret
            optimal_reward = env.optimal_reward()
            regrets[trial, step] = optimal_reward - reward
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for level in confidence_levels:
        confidence_intervals[level] = compute_confidence_interval(regrets, level)
    
    return regrets, confidence_intervals

def main():
    # Load configuration
    config = load_config()
    
    # Set random seeds
    np.random.seed(config['seeds']['numpy'])
    
    # Create Bernoulli environment
    bernoulli_env = BernoulliBandit(probs=config['environments']['bernoulli']['probs'])
    
    # Create LLM agent
    agent = LLMAgent(model="gpt-4")  # Will automatically read API key from llm_api.txt
    
    # Run simulation
    print("Running Bernoulli environment simulation with LLM agent...")
    regrets, intervals = run_simulation(
        bernoulli_env, agent, config['simulation']['n_steps'],
        config['simulation']['n_trials'], config['simulation']['confidence_levels']
    )
    
    # Plot results
    plot_regret_with_confidence(
        [agent], regrets, intervals,
        config, "Bernoulli"
    )

    # Create environment
    env = GaussianBandit(n_actions=10)

    # Generate configuration
    means, stds = generate_configuration(n_actions=10)

    # Set the environment parameters
    env.set(means, stds)

    # Use the environment
    reward = env.pull(0)  # Pull first arm and .....

if __name__ == "__main__":
    main()

# Add seeder for the bandits algorithms
#Now do a confidence interval 99% for the regret
# Create a folder called agents and save the agents in there and their strategies , and __init__.py file to import the agents add base agent  
# create a folder for the plots and save the plots in there
# folder for configuration and save the configuration in there 

