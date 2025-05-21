import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl

# Helper to get base agent name for color mapping
# (e.g., EpsilonGreedy from EpsilonGreedy(epsilon=0.1, bernoulli))
def get_base_agent_name(agent):
    return getattr(agent, '_name', str(agent).split('(')[0])

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
            'GradientBandit': {'color': '#9467bd', 'linestyle': '-', 'linewidth': 2},  # Purple color for gradient bandit
            # LLM agents (red dotted lines)
            'LLM': {'color': 'red', 'linestyle': ':', 'linewidth': 2.5},
            'LLMV2': {'color': 'red', 'linestyle': ':', 'linewidth': 2.5},
            # Gaussian variants (solid lines with same colors as base agents)
            'GaussianEpsilonGreedy': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2},
            'GaussianUCB': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 2},
            'GaussianThompsonSampling': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2},
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
                # Force red color and dotted line for all LLM agents
                style = {'color': 'red', 'linestyle': ':', 'linewidth': 2.5}
            
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
        plt.title(f'Average Cumulative Regret with Upper Confidence Intervals - {env_name} Environment', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Create plots directory if it doesn't exist
        plots_dir = config['paths']['plots_dir']
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