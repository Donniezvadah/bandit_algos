import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import importlib.util
from typing import Dict, List, Tuple, Any
import logging
import json
from datetime import datetime

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
from utils.confidence import compute_confidence_interval

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def sample_means(n_arms: int, easy_range: tuple = (0.1, 0.3), hard_range: tuple = (0.7, 0.9), 
                seed: int = None) -> np.ndarray:
    """Sample means from easy and hard ranges.
    
    Args:
        n_arms: Number of arms in the bandit
        easy_range: Tuple of (min, max) for easy arms (lower rewards)
        hard_range: Tuple of (min, max) for hard arms (higher rewards)
        seed: Random seed for reproducibility
        
    Returns:
        Array of means for each arm
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Randomly assign each arm to either easy or hard range
    is_hard = np.random.choice([True, False], size=n_arms, p=[0.5, 0.5])
    
    # Initialize means array
    means = np.zeros(n_arms)
    
    # Sample from appropriate range for each arm
    for i in range(n_arms):
        if is_hard[i]:
            means[i] = np.random.uniform(hard_range[0], hard_range[1])
        else:
            means[i] = np.random.uniform(easy_range[0], easy_range[1])
    
    return means

def setup_environment(env_config):
    """Initialize bandit environment based on configuration."""
    env_type = env_config['name']
    seed = env_config.get('seed', 42)
    
    if seed is not None:
        np.random.seed(seed)
    
    if env_type == 'bernoulli':
        # Sample means from easy and hard ranges
        n_arms = env_config.get('n_arms', 5)  # Default to 5 arms if not specified
        means = sample_means(n_arms, 
                           easy_range=(0.1, 0.3),
                           hard_range=(0.7, 0.9),
                           seed=seed)
        
        # Convert means to probabilities (clipping to [0.01, 0.99] for numerical stability)
        probs = np.clip(means, 0.01, 0.99)
        
        return BernoulliBandit(
            n_actions=n_arms,
            probs=probs.tolist()
        )
    elif env_type == 'gaussian':
        # Sample means from easy and hard ranges
        n_arms = env_config.get('n_arms', 5)  # Default to 5 arms if not specified
        means = sample_means(n_arms,
                           easy_range=(0.1, 0.3),
                           hard_range=(0.7, 0.9),
                           seed=seed)
        
        # Use small, constant standard deviation for all arms
        stds = np.ones(n_arms) * 0.1
        
        env = GaussianBandit(n_actions=n_arms)
        env.set(means.tolist(), stds.tolist())
        return env
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
    
    # Add LLM agent if enabled in config
    if any(agent['name'] == 'llm_agent' for agent in config.get('agents', [])):
        try:
            llm_agent = LLMAgent(
                model="gpt-4.1-nano",
                temperature=0.3,
                max_retries=3,
                timeout=30
            )
            agents.append(llm_agent)
        except Exception as e:
            print(f"Warning: Could not initialize LLM agent: {e}")
    
    # Add other agents with optimized parameters
    agent_classes = {
        'epsilon_greedy': EpsilonGreedy,
        'thompson_sampling': ThompsonSampling,
        'ucb': UCB,
        'gradient_bandit': GradientBandit,
        'llm_agent': LLMAgent
    }
    
    for agent_config in config.get('agents', []):
        agent_name = agent_config['name']
        if agent_name in agent_classes:
            agent_class = agent_classes[agent_name]
            agent_params = {k: v for k, v in agent_config.items() if k != 'name'}
            
            # Initialize agent with optimized parameters
            try:
                if agent_name == 'epsilon_greedy':
                    # Use decaying epsilon for better performance
                    agent = agent_class(epsilon=0.1, environment_type=env_type)
                elif agent_name == 'ucb':
                    # Use UCB1 with optimized exploration parameter
                    agent = agent_class(c=2.0)
                elif agent_name == 'gradient_bandit':
                    # Use gradient bandit with baseline and optimized learning rate
                    agent = agent_class(alpha=0.1, baseline=True)
                elif agent_name == 'llm_agent':
                    # LLM agent has already been added
                    continue
                else:
                    agent = agent_class(**agent_params)
            except Exception as e:
                print(f"Warning: Could not initialize agent {agent_name}: {e}")
                continue
                
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

class SimulationRunner:
    """Class to handle running bandit simulations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulation runner.
        
        Args:
            config: Configuration dictionary containing simulation parameters
        """
        self.config = config
        self.results_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_simulation(self, env: Any, agent: Any, n_steps: int, n_trials: int, 
                      confidence_levels: List[float]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run a single simulation with the given agent and environment.
        
        Args:
            env: The bandit environment
            agent: The agent to run
            n_steps: Number of steps per trial
            n_trials: Number of trials to run
            confidence_levels: List of confidence levels for intervals
            
        Returns:
            Tuple of (cumulative_regrets, confidence_intervals)
        """
        logger.info(f"Starting simulation for {agent.name}...")
        regrets = np.zeros((n_trials, n_steps))
        cumulative_regrets = np.zeros((n_trials, n_steps))
        
        for trial in range(n_trials):
            logger.info(f"Trial {trial + 1}/{n_trials}")
            
            # Reset environment and agent
            env.reset()
            agent.init_actions(env.action_count)
            
            # Run simulation
            for step in range(n_steps):
                action = agent.get_action()
                reward = env.pull(action)
                agent.update(action, reward)
                
                # Calculate regret
                optimal_reward = env.optimal_reward()
                regrets[trial, step] = optimal_reward - reward
                cumulative_regrets[trial, step] = (cumulative_regrets[trial, step-1] + regrets[trial, step] 
                                                 if step > 0 else regrets[trial, step])
        
        # Calculate confidence intervals
        logger.info("Computing confidence intervals...")
        confidence_intervals = {}
        for level in confidence_levels:
            logger.info(f"Computing {level*100}% confidence interval...")
            ci = compute_confidence_interval(cumulative_regrets, level)
            confidence_intervals.update(ci)
        
        return cumulative_regrets, confidence_intervals
    
    def run_bernoulli_experiments(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Run experiments with Bernoulli bandit environment."""
        logger.info("Starting Bernoulli experiments...")
        
        # Initialize environment using the new configuration structure
        env_config = self.config['environment']['bernoulli']
        env = setup_environment(env_config)
        
        # Initialize agents using the new configuration structure
        agents = setup_agents(env, self.config)
        
        # Run simulations
        all_regrets = {}
        all_intervals = {}
        
        for agent in agents:
            try:
                regrets, intervals = self.run_simulation(
                    env, agent,
                    self.config['experiment']['n_steps'],
                    self.config['experiment']['n_runs'],
                    self.config.get('confidence_levels', [0.95])
                )
                all_regrets[agent.name] = regrets
                all_intervals[agent.name] = intervals
                
                # Save results
                self._save_results(agent.name, regrets, intervals, "bernoulli")
                
            except Exception as e:
                logger.error(f"Error in simulation for {agent.name}: {str(e)}", exc_info=True)
                continue
        
        return all_regrets, all_intervals
    
    def run_gaussian_experiments(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Run experiments with Gaussian bandit environment."""
        logger.info("Starting Gaussian experiments...")
        
        # Initialize environment using the new configuration structure
        env_config = self.config['environment']['gaussian']
        env = setup_environment(env_config)
        
        # Initialize agents using the new configuration structure
        agents = setup_agents(env, self.config)
        
        # Run simulations
        all_regrets = {}
        all_intervals = {}
        
        for agent in agents:
            try:
                regrets, intervals = self.run_simulation(
                    env, agent,
                    self.config['experiment']['n_steps'],
                    self.config['experiment']['n_runs'],
                    self.config.get('confidence_levels', [0.95])
                )
                all_regrets[agent.name] = regrets
                all_intervals[agent.name] = intervals
                
                # Save results
                self._save_results(agent.name, regrets, intervals, "gaussian")
                
            except Exception as e:
                logger.error(f"Error in simulation for {agent.name}: {str(e)}", exc_info=True)
                continue
        
        return all_regrets, all_intervals
    
    def _save_results(self, agent_name: str, regrets: np.ndarray, 
                     intervals: Dict[str, Any], env_type: str) -> None:
        """Save simulation results to disk."""
        agent_dir = self.results_dir / env_type / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Save regrets
        np.save(agent_dir / "regrets.npy", regrets)
        
        # Save confidence intervals
        with open(agent_dir / "confidence_intervals.json", 'w') as f:
            json.dump(intervals, f, indent=2)
        
        # Save configuration
        with open(agent_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

def main():
    """Main function to run all experiments."""
    try:
        logger.info("Starting experiments...")
        
        # Load configuration
        logger.info("Loading configuration...")
        from omegaconf import OmegaConf
        config = OmegaConf.load('configurations/config.yaml')
        config = OmegaConf.to_container(config, resolve=True)
        logger.info(f"Configuration loaded: {config}")
        
        # Set random seed
        np.random.seed(config['seeds']['numpy'])
        logger.info(f"Set random seed to {config['seeds']['numpy']}")
        
        # Initialize runner and run experiments
        logger.info("Initializing SimulationRunner...")
        runner = SimulationRunner(config)
        
        # Run Bernoulli experiments
        logger.info("Starting Bernoulli experiments...")
        bernoulli_regrets, bernoulli_intervals = runner.run_bernoulli_experiments()
        logger.info("Bernoulli experiments completed, plotting results...")
        plot_regret_with_confidence(
            list(bernoulli_regrets.keys()),
            bernoulli_regrets,
            bernoulli_intervals,
            config,
            "Bernoulli"
        )
        
        # Run Gaussian experiments
        logger.info("Starting Gaussian experiments...")
        gaussian_regrets, gaussian_intervals = runner.run_gaussian_experiments()
        logger.info("Gaussian experiments completed, plotting results...")
        plot_regret_with_confidence(
            list(gaussian_regrets.keys()),
            gaussian_regrets,
            gaussian_intervals,
            config,
            "Gaussian"
        )
        
        logger.info("All experiments completed successfully!")
        
    except Exception as e:
        logger.error("Error in main function", exc_info=True)
        raise

if __name__ == "__main__":
    main()
