import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl

# Helper to get base agent name for color mapping
# (e.g., EpsilonGreedy from EpsilonGreedy(epsilon=0.1, bernoulli))
def get_base_agent_name(agent):
    return getattr(agent, '_name', str(agent).split('(')[0])

def plot_regret_with_confidence(agents, regrets, intervals, config, env_type):
    """Plot regret with confidence intervals for all agents."""
    plt.figure(figsize=(12, 8))
    for agent in agents:
        agent_name = agent.name
        print(f"Plotting for agent: {agent_name}")
        print(f"Regrets shape: {regrets[agent_name].shape}")
        print(f"Intervals keys: {intervals[agent_name].keys()}")
        mean_regret = np.mean(regrets[agent_name], axis=0)
        std_regret = np.std(regrets[agent_name], axis=0)
        steps = np.arange(1, len(mean_regret) + 1)
        plt.plot(steps, mean_regret, label=agent_name)
        plt.fill_between(steps, mean_regret - std_regret, mean_regret + std_regret, alpha=0.2)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title(f'Cumulative Regret over Time ({env_type} Environment)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(config['output']['plots_dir'], f'regret_{env_type.lower()}.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

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
    try:
        plt.figure(figsize=(12, 8))
        
        # Define colors and line styles for different agent types
        agent_styles = {
            # Base agents
            'EpsilonGreedy': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2},
            'UCB': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 2},
            'ThompsonSampling': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2},
            'KL-UCB': {'color': '#d62728', 'linestyle': '-', 'linewidth': 2},
            'GradientBandit': {'color': '#9467bd', 'linestyle': '-', 'linewidth': 2},
            # LLM agents (red dotted lines)
            'LLM': {'color': 'red', 'linestyle': ':', 'linewidth': 2.5},
        }
        
        # Fallback style for unknown agents
        default_style = {'color': '#7f7f7f', 'linestyle': '-', 'linewidth': 1.5}
        
        # Create a set to track which agent names we've already added to the legend
        legend_handles = {}
        
        for agent in agents:
            print(f"\nPlotting data for {agent.name}...")
            base_name = get_base_agent_name(agent)
            
            # Get style for this agent, or use default if not found
            style = agent_styles.get(base_name, default_style)
            
            # For LLM agents, use red dotted style
            if 'LLM' in agent.name or 'llm' in agent.name.lower():
                style = agent_styles.get('LLM', default_style)
            
            avg_regret = np.mean(regret[agent.name], axis=0)
            print(f"Average regret shape: {avg_regret.shape}")
            
            # Plot the average regret curve
            print(f"Plotting average regret curve for {agent.name}")
            line, = plt.plot(avg_regret, 
                          label=agent.name,
                          color=style['color'],
                          linestyle=style['linestyle'],
                          linewidth=style['linewidth'])
            
            # Store the first line of each base type for the legend
            if base_name not in legend_handles:
                legend_handles[base_name] = line
            
            # Plot confidence intervals with matching style but lighter color
            print(f"Plotting confidence intervals for {agent.name}")
            print(f"Available confidence levels: {confidence_intervals[agent.name].keys()}")
            for level, (lower, upper) in confidence_intervals[agent.name].items():
                print(f"Plotting {level} confidence interval")
                plt.fill_between(range(len(upper)), lower, upper, 
                               color=style['color'], 
                               alpha=0.1, 
                               linewidth=0)
                
        # Create a custom legend with one entry per agent type
        plt.legend(handles=[(h, plt.Line2D([0], [0], color=h.get_color(), 
                                         linestyle=h.get_linestyle(),
                                         linewidth=h.get_linewidth())) 
                          for h in legend_handles.values()],
                 labels=legend_handles.keys(),
                 loc='upper left',
                 fontsize=10)
        
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Cumulative Regret', fontsize=12)
        plt.title(f'Average Cumulative Regret with Confidence Intervals - {env_name} Environment', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Create plots directory if it doesn't exist
        plots_dir = config['output']['plots_dir']
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save plots with environment name in filename
        base_filename = f"regret_with_ci_{env_name.lower()}"
        print(f"Saving plots to {plots_dir}")
        plt.savefig(os.path.join(plots_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, f"{base_filename}.pdf"), bbox_inches='tight')
        plt.close()
        print("Plotting completed successfully")
        
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {e.__dict__}")
        raise 