import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import importlib.util

# Add project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environments.bernoulli_bandit import BernoulliBandit
from environments.gaussian_bandit import GaussianBandit
from agents.epsilon import EpsilonGreedyAgent as EpsilonGreedy
from agents.ucb import UCBAgent as UCB
from agents.thompson import ThompsonSamplingAgent as ThompsonSampling
from agents.gradient_bandit import GradientBanditAgent as GradientBandit
from agents.llm_agent import LLMAgent
from plots.plot_utils import plot_regret_with_confidence

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_environment(env_config):
    """Initialize bandit environment based on configuration."""
    env_type = env_config['name']
    seed = env_config.get('seed', 42)
    
    if env_type == 'bernoulli':
        if seed is not None:
            np.random.seed(seed)
        return BernoulliBandit(
            probs=env_config['probabilities'],
            n_actions=len(env_config['probabilities'])
        )
    elif env_type == 'gaussian':
        # For Gaussian, we'll create a custom configuration with the specified means and stds
        class CustomGaussianBandit(GaussianBandit):
            def __init__(self, means, stds, seed=None):
                self.means = np.array(means)
                self.stds = np.array(stds)
                self.action_count = len(means)
                self.seed = seed
                if seed is not None:
                    np.random.seed(seed)
                self.reset()
                
            def reset(self):
                self.best_action = np.argmax(self.means)
                self.best_mean = self.means[self.best_action]
                
            def pull(self, action):
                return np.random.normal(self.means[action], self.stds[action])
                
            def optimal_reward(self):
                return self.best_mean
                
        return CustomGaussianBandit(
            means=env_config['means'],
            stds=env_config['stds'],
            seed=seed
        )
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")

def setup_agents(env, config):
    """Initialize agents based on configuration."""
    agents = []
    
    # Determine the number of arms and environment type
    if hasattr(env, 'means'):  # Gaussian environment
        n_arms = len(env.means)
        env_type = 'gaussian'
    elif hasattr(env, '_probs'):  # Bernoulli environment
        n_arms = len(env._probs)
        env_type = 'bernoulli'
    else:
        n_arms = getattr(env, 'n_actions', 2)  # Default to 2 arms if unknown
        env_type = 'bernoulli'  # Default to bernoulli if unknown
    
    # Add LLM agent (always include)
    llm_agent = LLMAgent(
        model="gpt-4.1-nano",  # or whatever model you want to use
        temperature=0.3,
        max_retries=3,
        timeout=30
    )
    agents.append(llm_agent)
    
    # Add other agents
    agent_classes = {
        'epsilon_greedy': EpsilonGreedy,
        'thompson_sampling': ThompsonSampling,
        'ucb': UCB,
        'gradient_bandit': GradientBandit
    }
    
    for agent_config in config.get('agents', []):
        agent_name = agent_config['name']
        if agent_name in agent_classes:
            agent_class = agent_classes[agent_name]
            agent_params = {k: v for k, v in agent_config.items() if k != 'name'}
            
            # Initialize agent with appropriate parameters
            if agent_name == 'epsilon_greedy':
                agent = agent_class(epsilon=agent_params.get('epsilon', 0.1), 
                                  environment_type=env_type)
            elif agent_name == 'ucb':
                agent = agent_class()  # UCB doesn't take parameters in its constructor
            elif agent_name == 'gradient_bandit':
                # GradientBandit uses 'alpha' instead of 'step_size'
                alpha = agent_params.get('step_size', 0.1)
                agent = agent_class(alpha=alpha)
            else:
                agent = agent_class(**agent_params)
                
            # Initialize the agent's actions if needed
            if hasattr(agent, 'init_actions'):
                agent.init_actions(n_arms)
                
            agents.append(agent)
    
    return agents

def run_experiment(env, agents, n_steps, n_runs, seed=None):
    """Run the bandit experiment."""
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize results storage
    results = {
        'regret': {agent.name: np.zeros(n_steps) for agent in agents},
        'cumulative_regret': {agent.name: np.zeros(n_steps) for agent in agents},
        'confidence_intervals': {agent.name: np.zeros((2, n_steps)) for agent in agents}
    }
    
    # Store per-run regret for confidence intervals
    run_regrets = {agent.name: np.zeros((n_runs, n_steps)) for agent in agents}
    
    for run in tqdm(range(n_runs), desc="Running experiments"):
        # Reset environment and agents
        env.reset()
        for agent in agents:
            agent.reset()
            # Ensure the agent is properly initialized with the correct number of actions
            if hasattr(env, 'means'):  # Gaussian environment
                n_actions = len(env.means)
            elif hasattr(env, '_probs'):  # Bernoulli environment
                n_actions = len(env._probs)
            else:
                n_actions = getattr(env, 'n_actions', 2)  # Default to 2 actions if unknown
            
            # Initialize the agent with the correct number of actions
            if hasattr(agent, 'init_actions'):
                agent.init_actions(n_actions)
        
        cumulative_regret = 0
        
        # Run the experiment
        for step in range(n_steps):
            for agent in agents:
                action = agent.choose_action()
                reward = env.pull(action)
                agent.update(action, reward)
                
                # Calculate regret for this step
                if hasattr(env, 'means'):  # Gaussian environment
                    optimal_reward = max(env.means)
                    step_regret = optimal_reward - env.means[action]
                elif hasattr(env, '_probs'):  # Bernoulli environment
                    optimal_reward = max(env._probs)
                    step_regret = optimal_reward - env._probs[action]
                else:
                    n_arms = env.n_actions
                    optimal_reward = max([env.step(i) for i in range(n_arms)])
                    step_regret = optimal_reward - reward
                
                cumulative_regret += step_regret
                run_regrets[agent.name][run, step] = cumulative_regret
    
    # Calculate mean and confidence intervals across runs
    for agent in agents:
        name = agent.name
        # Calculate mean cumulative regret
        results['cumulative_regret'][name] = np.mean(run_regrets[name], axis=0)
        # Calculate standard error of the mean
        std_err = np.std(run_regrets[name], axis=0) / np.sqrt(n_runs)
        # 95% confidence interval
        results['confidence_intervals'][name][0] = results['cumulative_regret'][name] - 1.96 * std_err
        results['confidence_intervals'][name][1] = results['cumulative_regret'][name] + 1.96 * std_err
        # Per-step regret (average across runs)
        results['regret'][name] = np.mean(np.diff(run_regrets[name], axis=1, prepend=0), axis=0)
    
    return results


def plot_scenario_results(results, agents, config, scenario_name, output_dir):
    """Plot and save results for a single scenario."""
    # Set style for better looking plots
    plt.style.use('ggplot')
    
    # Create a single figure
    plt.figure(figsize=(10, 6))
    
    # Define a clean color palette
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ]
    
    # Plot each agent's cumulative regret
    for i, agent in enumerate(agents):
        name = agent.name
        if name not in results['cumulative_regret']:
            print(f"Warning: No results found for agent {name}")
            continue
            
        # Get the cumulative regret data
        cum_regret = results['cumulative_regret'][name]
        
        # Plot the cumulative regret with thinner lines but keeping markers visible
        plt.plot(
            cum_regret,
            label=name,
            color=colors[i % len(colors)],
            linewidth=1.2,  # Thinner line
            alpha=0.9,
            marker='o',
            markersize=4,  # Keep marker size the same
            markevery=max(1, len(cum_regret) // 20),  # Show some markers for clarity
            markeredgewidth=1,  # Keep marker edge thin
            markerfacecolor=colors[i % len(colors)],  # Match line color
            markeredgecolor='none'  # No edge on markers for cleaner look
        )
    
    # Customize the plot
    title = f"{scenario_name.replace('_', ' ').title()} Environment"
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Time Step', fontsize=12, labelpad=10)
    plt.ylabel('Cumulative Regret', fontsize=12, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, framealpha=1, edgecolor='black', loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    scenario_name_clean = scenario_name.lower().replace(' ', '_')
    output_path = os.path.join(output_dir, f'{scenario_name_clean}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Saved plot to {output_path}")


def main():
    # Load configuration
    config_path = "configurations/experiment/experiment.yaml"
    config = load_config(config_path)
    
    # Set random seeds for reproducibility
    np.random.seed(config['experiment']['seed'])
    
    # Output directory for plots
    output_dir = os.path.join("plots", "scenarios")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiments for each scenario
    for scenario_name, env_config in config['environments'].items():
        print(f"\n{'='*50}")
        print(f"Running scenario: {scenario_name}")
        print(f"{'='*50}")
        
        # Setup environment and agents
        env = setup_environment(env_config)
        agents = setup_agents(env, config)
        
        # Run experiment
        results = run_experiment(
            env,
            agents,
            n_steps=config['experiment']['n_steps'],
            n_runs=config['experiment']['n_runs'],
            seed=config['experiment']['seed']
        )
        
        # Plot results
        plot_scenario_results(results, agents, config, scenario_name, output_dir)

if __name__ == "__main__":
    main()
