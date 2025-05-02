import matplotlib.pyplot as plt
import os
import numpy as np

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
        
        # Define colors for each agent
        colors = {
            'EpsilonGreedy': 'blue',
            'UCB': 'green',
            'ThompsonSampling': 'red',
            'LLM': 'purple'
        }
        
        for agent in agents:
            print(f"\nPlotting data for {agent.name}...")
            agent_color = colors.get(agent.name, 'black')
            
            # Calculate average regret across trials
            avg_regret = np.mean(regret[agent.name], axis=0)
            print(f"Average regret shape: {avg_regret.shape}")
            
            # Plot the average regret curve
            print(f"Plotting average regret curve for {agent.name}")
            plt.plot(avg_regret, label=agent.name, linewidth=2, color=agent_color)
            
            # Plot only upper confidence interval
            print(f"Plotting confidence intervals for {agent.name}")
            print(f"Available confidence levels: {confidence_intervals[agent.name].keys()}")
            for level, (lower, upper) in confidence_intervals[agent.name].items():
                print(f"Plotting {level} upper confidence interval")
                plt.plot(upper, '--', color=agent_color, alpha=0.5, label=f'{agent.name} {level} Upper CI')
        
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Cumulative Regret', fontsize=12)
        plt.title(f'Average Regret with Upper Confidence Intervals - {env_name} Environment', fontsize=14)
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