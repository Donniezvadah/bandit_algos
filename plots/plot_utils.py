import matplotlib.pyplot as plt
import os

def plot_regret_with_confidence(agents, regret, confidence_intervals, config, env_name):
    """
    Plot regret with confidence intervals and save to plots directory.
    
    Args:
        agents: List of agents
        regret: Dictionary of regret data
        confidence_intervals: Dictionary of confidence intervals
        config: Configuration dictionary
        env_name: Name of the environment (for file naming)
    """
    plt.figure(figsize=(12, 8))
    
    # Define colors for each agent
    colors = {
        'EpsilonGreedy': 'blue',
        'UCB': 'green',
        'ThompsonSampling': 'red',
        'LLM': 'purple'
    }
    
    for agent in agents:
        agent_color = colors.get(agent.name, 'black')
        # Plot the regret curve
        plt.plot(regret[agent.name], label=agent.name, linewidth=2, color=agent_color)
        
        # Plot only upper confidence bound with matching color
        _, upper = confidence_intervals[agent.name]["95%"]
        plt.plot(upper, '--', color=agent_color, alpha=0.5, label=f'{agent.name} 95% Upper CI')
    
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.title(f'Regret with 95% Upper Confidence Bound - {env_name} Environment', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Create plots directory if it doesn't exist
    plots_dir = config['paths']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save plots with environment name in filename
    base_filename = f"regret_with_upper_ci_{env_name.lower()}"
    plt.savefig(os.path.join(plots_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, f"{base_filename}.pdf"), bbox_inches='tight')
    plt.close() 