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
import os
import sys
from omegaconf import OmegaConf
from agents.gaussian_epsilon_greedy import GaussianEpsilonGreedyAgent
from agents.gaussian_ucb import GaussianUCBAgent
from agents.gaussian_thompson_sampling import GaussianThompsonSamplingAgent
from agents.ucb_kl import KLUCBAgent

from agents.gradient_bandit import GradientBanditAgent

def load_config() -> tuple[dict, dict, dict]:
    """
    Load configuration from YAML files using OmegaConf.
    
    Returns:
        tuple: (merged_config, bernoulli_env_cfg, gaussian_env_cfg)
            - merged_config: Merged configuration dictionary
            - bernoulli_env_cfg: Bernoulli environment configuration
            - gaussian_env_cfg: Gaussian environment configuration
    """
    try:
        config_files = [
            'configurations/config.yaml',
            'configurations/experiment/experiment.yaml',
            'configurations/agent/epsilon_greedy.yaml',
            'configurations/environment/bernoulli_env.yaml',
            'configurations/environment/gaussian_env.yaml'
        ]
        
        configs = []
        for file in config_files:
            try:
                cfg = OmegaConf.load(file)
                configs.append(cfg)
            except FileNotFoundError:
                print(f"Warning: Configuration file {file} not found")
                configs.append(OmegaConf.create())
            except Exception as e:
                print(f"Error loading configuration file {file}: {e}")
                raise
                
        # Merge all configs (default: Bernoulli)
        merged = OmegaConf.merge(*configs)
        merged_dict = OmegaConf.to_container(merged, resolve=True)
        
        # Validate required fields
        required_fields = ['experiment', 'environment', 'agent']
        for field in required_fields:
            if field not in merged_dict:
                raise ValueError(f"Missing required configuration field: {field}")
                
        return merged_dict, configs[3], configs[4]
        
    except Exception as e:
        print(f"Error loading YAML configuration: {e}")
        sys.exit(1)

def run_simulation(
    env, agent, n_steps: int, n_trials: int, confidence_levels: list[float]
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Run the bandit simulation with a single agent.
    
    Args:
        env: The bandit environment
        agent: The bandit agent
        n_steps: Number of steps per trial
        n_trials: Number of trials to run
        confidence_levels: List of confidence levels for intervals
        
    Returns:
        tuple: (regrets, cumulative_regrets, regret_intervals, cumulative_regret_intervals)
            - regrets: Array of regrets for each trial and step
            - cumulative_regrets: Array of cumulative regrets for each trial and step
            - regret_intervals: List of confidence intervals for regrets
            - cumulative_regret_intervals: List of confidence intervals for cumulative regrets
    """
    if not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")
    if not all(0 < cl <= 1 for cl in confidence_levels):
        raise ValueError("Confidence levels must be between 0 and 1")
    
    print(f"\nStarting simulation for {agent.name}...")
    regrets = np.zeros((n_trials, n_steps))
    cumulative_regrets = np.zeros((n_trials, n_steps))
    
    for trial in range(n_trials):
        try:
            print(f"\nTrial {trial + 1}/{n_trials}")
            env.reset()
            agent.reset()
            
            for t in range(n_steps):
                action = agent.get_action()
                reward = env.step(action)
                agent.update(action, reward)
                
                # Calculate regret
                optimal_reward = env.max_reward()
                regrets[trial, t] = optimal_reward - reward
                if t > 0:
                    cumulative_regrets[trial, t] = cumulative_regrets[trial, t-1] + regrets[trial, t]
                else:
                    cumulative_regrets[trial, t] = regrets[trial, t]
                    
            # Print trial summary
            avg_regret = np.mean(regrets[trial])
            avg_cumulative_regret = np.mean(cumulative_regrets[trial])
            print(f"Trial {trial + 1} complete. Average regret: {avg_regret:.4f}, Average cumulative regret: {avg_cumulative_regret:.4f}")
            
        except Exception as e:
            print(f"Error during trial {trial + 1}: {e}")
            continue
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
        config, bernoulli_env_cfg, gaussian_env_cfg = load_config()
        print("Configuration loaded successfully")
        
        # Set random seeds
        print("Setting random seeds...")
        np.random.seed(config['seeds']['numpy'])
        print(f"Random seed set to: {config['seeds']['numpy']}")
        
        # Test Bernoulli environment
        print("\nTesting Bernoulli environment with all agents...")
        probs = np.array([float(prob) for prob in bernoulli_env_cfg['probabilities']])
        print(f"Probabilities: {probs}")
        env = BernoulliBandit(n_actions=len(probs), probs=probs)
        print("Bernoulli environment initialized")
        # Initialize agents (after probs is defined)
        agents = [
            EpsilonGreedyAgent(epsilon=0.1, environment_type='bernoulli'),
            UCBAgent(),
            KLUCBAgent(n_arms=len(probs)),
            ThompsonSamplingAgent(environment_type='bernoulli'),
            GradientBanditAgent(alpha=0.1, baseline=True),
            LLMAgent(model="gpt-4.1-nano")
        ]
        print(f"Initialized {len(agents)} agents")
        
        # Run simulations for Bernoulli environment
        print("Starting Bernoulli simulations...")
        all_regrets_bernoulli = {}
        all_intervals_bernoulli = {}
        
        for agent in agents:
            print(f"\nTesting {agent.name}...")
            try:
                regrets, intervals = run_simulation(
                    env, agent, config['experiment']['n_steps'],
                    config['experiment']['n_runs'], config.get('confidence_levels', [0.95])
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
        means = np.array([float(mean) for mean in gaussian_env_cfg['means']])
        stds = np.array([float(std) for std in gaussian_env_cfg['stds']])
        env = GaussianBandit(n_actions=len(means))
        print(f"Means: {means}")
        print(f"Stds: {stds}")
        env.set(means, stds)
        print("Gaussian environment initialized")
        
        # Update agents for Gaussian environment
        agents = [
            GaussianEpsilonGreedyAgent(n_arms=len(means), epsilon=0.1),
            GaussianUCBAgent(n_arms=len(means)),
            GaussianThompsonSamplingAgent(n_arms=len(means)),
            GradientBanditAgent(alpha=0.1, baseline=True),
            LLMAgent(model="o4-mini")
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
                    env, agent, config['experiment']['n_steps'],
                    config['experiment']['n_runs'], config.get('confidence_levels', [0.95])
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

