# bandits/main.py
import numpy as np
import matplotlib.pyplot as plt
from environments.gaussian_bandit import GaussianBandit
from environments.bernoulli_bandit import BernoulliBandit
from agents.epsilon import EpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.thompson import ThompsonSamplingAgent
from agents.llm_agent import LLMAgent
from utils.confidence import compute_confidence_interval
from plots.plot_utils import plot_regret_with_confidence
import configparser
import xml.etree.ElementTree as ET
import os

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
            },
            'gaussian': {
                'means': [float(mean.text) for mean in root.findall('environments/gaussian/means/mean')],
                'stds': [float(std.text) for std in root.findall('environments/gaussian/stds/std')]
            }
        }
    }
    return config

def run_simulation(env, agents, n_steps, n_trials, confidence_levels):
    """Run the bandit simulation."""
    n_agents = len(agents)
    regrets = np.zeros((n_agents, n_trials, n_steps))
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Reset environment and agents
        env.reset()
        for agent in agents:
            agent.init_actions(env.action_count)
        
        # Run simulation
        for step in range(n_steps):
            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{n_steps}", end='\r')
            
            for i, agent in enumerate(agents): #remove for loop here and reduce complexity and simplify the def run_simulation (1 agent at a time)
                #print action and reward here 
                #Use epsilon = 0.1
                #50 steps
                
                # Get action from agent
                action = agent.get_action()
                
                # Get reward from environment
                reward = env.pull(action)
                
                # Update agent
                agent.update(action, reward)
                
                # Calculate regret
                optimal_reward = env.optimal_reward()
                regrets[i, trial, step] = optimal_reward - reward
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for level in confidence_levels:
        confidence_intervals[level] = [
            compute_confidence_interval(regrets[i], level)
            for i in range(n_agents)
        ]
    
    return regrets, confidence_intervals

def main():
    # Load configuration
    config = load_config()
    
    # Set random seeds
    np.random.seed(config['seeds']['numpy'])
    
    # Create environments
    bernoulli_env = BernoulliBandit(probs=config['environments']['bernoulli']['probs'])
    gaussian_env = GaussianBandit(
        means=config['environments']['gaussian']['means'],
        stds=config['environments']['gaussian']['stds']
    )
    
    # Create agents
    agents = [
        EpsilonGreedyAgent(epsilon=0.1),
        UCBAgent(),
        ThompsonSamplingAgent(),
        LLMAgent(api_key=os.getenv('OPENAI_API_KEY'))  # Add LLM agent
    ]
    
    # Run simulations
    print("Running Bernoulli environment simulation...")
    bernoulli_regrets, bernoulli_intervals = run_simulation(
        bernoulli_env, agents, config['simulation']['n_steps'],
        config['simulation']['n_trials'], config['simulation']['confidence_levels']
    )
    
    print("\nRunning Gaussian environment simulation...")
    gaussian_regrets, gaussian_intervals = run_simulation(
        gaussian_env, agents, config['simulation']['n_steps'],
        config['simulation']['n_trials'], config['simulation']['confidence_levels']
    )
    
    # Plot results
    plot_regret_with_confidence(
        agents, bernoulli_regrets, bernoulli_intervals,
        config, "Bernoulli"
    )
    
    plot_regret_with_confidence(
        agents, gaussian_regrets, gaussian_intervals,
        config, "Gaussian"
    )

if __name__ == "__main__":
    main()

# Add seeder for the bandits algorithms
#Now do a confidence interval 99% for the regret
# Create a folder called agents and save the agents in there and their strategies , and __init__.py file to import the agents add base agent  
# create a folder for the plots and save the plots in there
# folder for configuration and save the configuration in there 

