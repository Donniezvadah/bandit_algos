import numpy as np
from scipy import stats

def compute_confidence_interval(data, confidence_level=0.95):
    """
    Compute confidence intervals for the given data.
    
    Args:
        data (np.ndarray): Array of shape (n_trials, n_steps) containing the data.
        confidence_level (float): The confidence level (default: 0.95).
        
    Returns:
        dict: Dictionary with confidence level as key and (lower_bound, upper_bound) as value
    """
    # Compute mean and standard error across trials for each step
    mean = np.mean(data, axis=0)  # Shape: (n_steps,)
    std = np.std(data, axis=0)  # Shape: (n_steps,)
    n = data.shape[0]  # Number of trials
    se = std / np.sqrt(n)  # Standard error
    
    # Compute critical value for the given confidence level
    z = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Compute confidence intervals
    lower_bound = np.maximum(0, mean - z * se)  # Ensure non-negative
    upper_bound = mean + z * se
    
    return {f"{int(confidence_level*100)}%": (lower_bound, upper_bound)}

def compute_regret_confidence_intervals(regret_data, confidence_levels=[0.95]):
    """
    Compute confidence intervals for regret data.
    
    Args:
        regret_data (dict): Dictionary of regret data for each agent
        confidence_levels (list): List of confidence levels
        
    Returns:
        dict: Dictionary of confidence intervals for each agent
    """
    ci_data = {}
    
    for agent_name, regrets in regret_data.items():
        ci_data[agent_name] = {}
        for level in confidence_levels:
            ci_data[agent_name].update(compute_confidence_interval(regrets, level))
            
    return ci_data 