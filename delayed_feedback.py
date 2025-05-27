import os
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from agents.epsilon import EpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.thompson import ThompsonSamplingAgent
from agents.ucb_kl import KLUCBAgent
from agents.gaussian_epsilon_greedy import GaussianEpsilonGreedyAgent
from agents.gaussian_ucb import GaussianUCBAgent
from agents.gaussian_thompson_sampling import GaussianThompsonSamplingAgent
from agents.llm_agent import LLMAgent
from environments.bernoulli_bandit import BernoulliBandit
from environments.gaussian_bandit import GaussianBandit
import time
from typing import Dict, List, Any, Optional, Tuple

def get_llm_agent(n_arms: int, **kwargs) -> LLMAgent:
    """Initialize LLM agent with appropriate parameters."""
    try:
        agent = LLMAgent(
            model="gpt-4.1-nano",
            temperature=0.3,
            max_retries=3,
            timeout=30
        )
        agent.init_actions(n_arms)
        return agent
    except Exception as e:
        print(f"Error initializing LLM agent: {e}")
        raise

AGENT_CLASSES = {
    'epsilon_greedy': EpsilonGreedyAgent,
    'ucb': UCBAgent,
    'thompson_sampling': ThompsonSamplingAgent,
    'kl_ucb': KLUCBAgent,
    'gaussian_epsilon_greedy': GaussianEpsilonGreedyAgent,
    'gaussian_ucb': GaussianUCBAgent,
    'gaussian_thompson_sampling': GaussianThompsonSamplingAgent,
    'llm_agent': get_llm_agent,
}

ENV_CLASSES = {
    'bernoulli': BernoulliBandit,
    'gaussian': GaussianBandit,
}

def run_experiment(agent_names: List[str], env_name: str, env_args: Dict,
                  delays: List[int], n_arms: int, n_runs: int, 
                  horizon: int, seed: int) -> Dict[str, List[float]]:
    """Run the delayed feedback experiment with the specified agents and environment.
    
    Args:
        agent_names: List of agent names to run
        env_name: Name of the environment ('bernoulli' or 'gaussian')
        env_args: Arguments for the environment
        delays: List of delays to test
        n_arms: Number of arms in the bandit
        n_runs: Number of runs per delay
        horizon: Number of steps per run
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping agent names to lists of average regrets for each delay
    """
    results = {agent: [] for agent in agent_names}
    
    for agent_name in agent_names:
        agent_factory = AGENT_CLASSES[agent_name]
        
        for delay in delays:
            regrets_runs = []
            
            for run in range(n_runs):
                # Set random seed for reproducibility
                run_seed = seed + run
                np.random.seed(run_seed)
                
                # Initialize environment
                env = ENV_CLASSES[env_name](**env_args)
                
                # For Gaussian environment, set means and stds
                if env_name == 'gaussian' and hasattr(env, 'set'):
                    # Sample means from a range to ensure some arms are better than others
                    means = np.linspace(0.1, 0.9, n_arms)
                    stds = np.ones(n_arms) * 0.1  # Low variance for easier learning
                    env.set(means, stds)
                
                # Initialize agent
                if agent_name == 'llm_agent':
                    agent = agent_factory(n_arms)
                elif agent_name == 'epsilon_greedy':
                    agent = agent_factory(epsilon=0.1)
                else:
                    agent = agent_factory(n_arms)
                
                if hasattr(agent, 'init_actions'):
                    agent.init_actions(n_arms)
                if hasattr(agent, 'reset'):
                    agent.reset()
                
                feedback_queue = []
                rewards = [None] * horizon
                optimal = env.optimal_reward()
                regrets = [None] * horizon
                
                # Run the experiment for 'horizon' steps
                for t in range(horizon):
                    # Choose action
                    if hasattr(agent, 'get_action'):
                        action = agent.get_action()
                    elif hasattr(agent, 'select_action'):
                        action = agent.select_action()
                    else:
                        raise ValueError(f"Agent {agent_name} doesn't have a valid action selection method")
                    
                    # Add to feedback queue
                    feedback_queue.append((t, action))
                    
                    # Process delayed feedback if available
                    if t >= delay:
                        feedback_t, feedback_arm = feedback_queue[t - delay]
                        reward = env.pull(feedback_arm)
                        agent.update(feedback_arm, reward)
                        rewards[feedback_t] = reward
                        regrets[feedback_t] = optimal - reward
                
                # Process any remaining feedback
                for t in range(max(horizon - delay, 0), horizon):
                    feedback_t, feedback_arm = feedback_queue[t]
                    reward = env.pull(feedback_arm)
                    agent.update(feedback_arm, reward)
                    rewards[feedback_t] = reward
                    regrets[feedback_t] = optimal - reward
                
                # Store the total regret for this run
                total_regret = np.nansum(regrets)
                regrets_runs.append(total_regret)
            
            # Store the average regret across runs for this delay
            avg_regret = np.mean(regrets_runs)
            results[agent_name].append(avg_regret)
            
            print(f"{agent_name} (delay={delay}): Avg regret = {avg_regret:.2f}")
    
    return results

def make_latex_table(results, delays, agent_labels, filename):
    header = " & " + " & ".join(str(d) for d in delays) + " \\\\ \\midrule\n"
    rows = []
    for agent, label in agent_labels.items():
        vals = " & ".join(f"{int(r):,}" for r in results[agent])
        rows.append(f"{label} & {vals} \\\\")
    table = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\begin{tabular}{l" + "c" * len(delays) + "}\n\\toprule\n"
        "$\\delta$" + header +
        "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Influence of the delay: regret when the feedback is provided every $\\delta$ steps.}\n"
        "\\end{table}\n"
    )
    with open(filename, "w") as f:
        f.write(table)
    return table

def plot_regret_vs_delay(results, delays, agent_labels, filename):
    import matplotlib as mpl
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = mpl.colormaps['tab10']
    for idx, (agent, label) in enumerate(agent_labels.items()):
        ax.plot(delays, results[agent], label=label, linewidth=1.5, color=colors(idx), alpha=0.95)
    ax.set_xlabel('Delay ($\\delta$)', fontsize=12)
    ax.set_ylabel('Cumulative Regret', fontsize=12)
    ax.set_title('Regret vs Delay', fontsize=13, pad=12)
    ax.set_xscale('log')
    ax.legend(frameon=True, fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=11)
    fig.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

@hydra.main(config_path="../../configurations", config_name="delayed_feedback_experiment", version_base=None)
def main(cfg: DictConfig):
    exp_cfg = cfg['experiment']
    delays = exp_cfg['delays']
    n_arms = exp_cfg['n_arms']
    n_runs = exp_cfg['n_runs']
    horizon = exp_cfg['horizon']
    seed = exp_cfg['seed']
    output_dir = exp_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Bernoulli
    bernoulli_agents = cfg['bernoulli_agent_names']
    bernoulli_labels = {
        'epsilon_greedy': 'Epsilon-Greedy',
        'ucb': 'UCB',
        'thompson_sampling': 'Thompson Sampling',
        'kl_ucb': 'KL-UCB',
        'llm_agent': 'LLM Agent'
    }
    bernoulli_results = run_experiment(
        bernoulli_agents, 'bernoulli', {'n_actions': n_arms}, delays, n_arms, n_runs, horizon, seed
    )
    latex_bernoulli = make_latex_table(
        bernoulli_results, delays, bernoulli_labels, os.path.join(output_dir, "delayed_feedback_table_bernoulli.tex")
    )
    plot_regret_vs_delay(
        bernoulli_results, delays, bernoulli_labels, os.path.join(output_dir, "regret_vs_delay_bernoulli.png")
    )

    # Gaussian
    gaussian_agents = cfg['gaussian_agent_names']
    gaussian_labels = {
        'gaussian_epsilon_greedy': 'Gaussian Epsilon-Greedy',
        'gaussian_ucb': 'Gaussian UCB',
        'gaussian_thompson_sampling': 'Gaussian Thompson Sampling',
        'llm_agent': 'LLM Agent'
    }
    gaussian_results = run_experiment(
        gaussian_agents, 'gaussian', {'n_actions': n_arms}, delays, n_arms, n_runs, horizon, seed
    )
    latex_gaussian = make_latex_table(
        gaussian_results, delays, gaussian_labels, os.path.join(output_dir, "delayed_feedback_table_gaussian.tex")
    )
    plot_regret_vs_delay(
        gaussian_results, delays, gaussian_labels, os.path.join(output_dir, "regret_vs_delay_gaussian.png")
    )

    # Write README
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("# Delayed Feedback Experiments\n\n")
        f.write("## Bernoulli Bandit\n\n")
        f.write("![Bernoulli Regret](regret_vs_delay_bernoulli.png)\n\n")
        f.write("```latex\n" + latex_bernoulli + "```\n\n")
        f.write("## Gaussian Bandit\n\n")
        f.write("![Gaussian Regret](regret_vs_delay_gaussian.png)\n\n")
        f.write("```latex\n" + latex_gaussian + "```\n")

if __name__ == "__main__":
    main() 