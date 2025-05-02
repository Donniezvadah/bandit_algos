# main.py
import numpy as np
import matplotlib.pyplot as plt
from environments.bernoulli_bandit import BernoulliBandit
from environments.gaussian_bandit import GaussianBandit, generate_configuration
from agents.llm_agent import LLMAgent
from agents.epsilon import EpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.thompson import ThompsonSamplingAgent
from utils.confidence import compute_confidence_interval
from plots.plot_utils import plot_regret_with_confidence
import configparser
import xml.etree.ElementTree as ET
import os
import sys

def load_config():
    """Load configuration from XML file."""
    print("Loading configuration...")
    try:
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
        print("Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def run_simulation(env, agent, n_steps, n_trials, confidence_levels):
    """Run the bandit simulation with a single agent."""
    print(f"\nStarting simulation for {agent.name}...")
    regrets = np.zeros((n_trials, n_steps))
    cumulative_regrets = np.zeros((n_trials, n_steps))
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Reset environment and agent
        env.reset()
        agent.init_actions(env.action_count)
        
        # Run simulation
        for step in range(n_steps):
            # Get action from agent
            action = agent.get_action()
            
            # Get reward from environment
            reward = env.pull(action)
            
            # Print detailed debugging information
            print(f"Step {step + 1}/{n_steps}")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            if isinstance(agent, EpsilonGreedyAgent):
                print(f"Agent state - Successes: {agent._successes}, Failures: {agent._failures}")
            elif isinstance(agent, UCBAgent):
                print(f"Agent state - Rewards: {agent._rewards}, Counts: {agent._counts}")
            elif isinstance(agent, ThompsonSamplingAgent):
                print(f"Agent state - Successes: {agent._successes}, Failures: {agent._failures}")
            elif isinstance(agent, LLMAgent):
                print(f"Agent state - Rewards: {agent._rewards}, Counts: {agent._counts}")
            
            # Update agent
            agent.update(action, reward)
            
            # Calculate regret
            optimal_reward = env.optimal_reward()
            regrets[trial, step] = optimal_reward - reward
            if step == 0:
                cumulative_regrets[trial, step] = regrets[trial, step]
            else:
                cumulative_regrets[trial, step] = cumulative_regrets[trial, step-1] + regrets[trial, step]
    
    # Calculate confidence intervals
    print("Computing confidence intervals...")
    confidence_intervals = {}
    for level in confidence_levels:
        print(f"Computing {level*100}% confidence interval...")
        ci = compute_confidence_interval(cumulative_regrets, level)
        confidence_intervals.update(ci)
        print(f"Confidence interval for {level*100}%: {ci}")
    
    return cumulative_regrets, confidence_intervals

def main():
    print("Starting main function...")
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config()
        print("Configuration loaded successfully")
        
        # Set random seeds
        print("Setting random seeds...")
        np.random.seed(config['seeds']['numpy'])
        print(f"Random seed set to: {config['seeds']['numpy']}")
        
        # Initialize agents
        print("Initializing agents...")
        agents = [
            EpsilonGreedyAgent(epsilon=0.1, environment_type='bernoulli'),
            UCBAgent(),
            ThompsonSamplingAgent(environment_type='bernoulli'),
            LLMAgent(model="gpt-3.5-turbo")
        ]
        print(f"Initialized {len(agents)} agents")
        
        # Test Bernoulli environment
        print("\nTesting Bernoulli environment with all agents...")
        probs = np.array([float(prob) for prob in config['environments']['bernoulli']['probs']])
        print(f"Probabilities: {probs}")
        env = BernoulliBandit(n_actions=10, probs=probs)
        print("Bernoulli environment initialized")
        
        # Run simulations for Bernoulli environment
        print("Starting Bernoulli simulations...")
        all_regrets_bernoulli = {}
        all_intervals_bernoulli = {}
        
        for agent in agents:
            print(f"\nTesting {agent.name}...")
            try:
                regrets, intervals = run_simulation(
                    env, agent, config['simulation']['n_steps'],
                    config['simulation']['n_trials'], config['simulation']['confidence_levels']
                )
                all_regrets_bernoulli[agent.name] = regrets
                all_intervals_bernoulli[agent.name] = intervals
                print(f"Completed simulation for {agent.name}")
                print(f"Regrets shape: {regrets.shape}")
                print(f"Intervals keys: {intervals.keys()}")
            except Exception as e:
                print(f"Error in simulation for {agent.name}: {str(e)}")
                print(f"Error type: {type(e)}")
                continue
        
        # Plot Bernoulli results
        print("Plotting Bernoulli results...")
        try:
            plot_regret_with_confidence(
                agents, all_regrets_bernoulli, all_intervals_bernoulli,
                config, "Bernoulli"
            )
            print("Bernoulli plots saved successfully")
        except Exception as e:
            print(f"Error plotting Bernoulli results: {str(e)}")
        
        # Test Gaussian environment
        print("\nTesting Gaussian environment with all agents...")
        env = GaussianBandit(n_actions=10)
        means = np.array([float(mean) for mean in config['environments']['gaussian']['means']])
        stds = np.array([float(std) for std in config['environments']['gaussian']['stds']])
        print(f"Means: {means}")
        print(f"Stds: {stds}")
        env.set(means, stds)
        print("Gaussian environment initialized")
        
        # Update agents for Gaussian environment
        agents = [
            EpsilonGreedyAgent(epsilon=0.1, environment_type='gaussian'),
            UCBAgent(),
            ThompsonSamplingAgent(environment_type='gaussian'),
            LLMAgent(model="o3-mini")
        ]
        print(f"Updated agents for Gaussian environment")
        
        # Run simulations for Gaussian environment
        print("Starting Gaussian simulations...")
        all_regrets_gaussian = {}
        all_intervals_gaussian = {}
        
        for agent in agents:
            print(f"\nTesting {agent.name}...")
            try:
                regrets, intervals = run_simulation(
                    env, agent, config['simulation']['n_steps'],
                    config['simulation']['n_trials'], config['simulation']['confidence_levels']
                )
                all_regrets_gaussian[agent.name] = regrets
                all_intervals_gaussian[agent.name] = intervals
                print(f"Completed simulation for {agent.name}")
                print(f"Regrets shape: {regrets.shape}")
                print(f"Intervals keys: {intervals.keys()}")
            except Exception as e:
                print(f"Error in simulation for {agent.name}: {str(e)}")
                print(f"Error type: {type(e)}")
                continue
        
        # Plot Gaussian results
        print("Plotting Gaussian results...")
        try:
            plot_regret_with_confidence(
                agents, all_regrets_gaussian, all_intervals_gaussian,
                config, "Gaussian"
            )
            print("Gaussian plots saved successfully")
        except Exception as e:
            print(f"Error plotting Gaussian results: {str(e)}")
        
        print("Done!")
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {e.__dict__}")
        raise

if __name__ == "__main__":
    main()

# Add seeder for the bandits algorithms
#Now do a confidence interval 99% for the regret
# Create a folder called agents and save the agents in there and their strategies , and __init__.py file to import the agents add base agent  
# create a folder for the plots and save the plots in there
# folder for configuration and save the configuration in there 

