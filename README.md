# Multi-Armed Bandit Algorithms Framework

A comprehensive, production-grade framework for implementing, testing, and comparing various multi-armed bandit (MAB) algorithms in both stationary and non-stationary environments. This project provides robust implementations of classic bandit algorithms like ε-greedy, UCB variants, and Thompson Sampling, along with cutting-edge techniques including KL-UCB and LLM-based agents. The framework is designed for both research and production use, with a focus on modularity, extensibility, and performance.

## Table of Contents
1. [Introduction](#introduction)
2. [Algorithms](#algorithms)
3. [Environments](#environments)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Configuration](#configuration)
8. [Advanced Features](#advanced-features)
9. [Results and Visualization](#results-and-visualization)
10. [Performance Considerations](#performance-considerations)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)
13. [Frequently Asked Questions](#frequently-asked-questions)
14. [Contributing](#contributing)
15. [License](#license)
16. [Acknowledgements](#acknowledgements)
17. [References](#references)

## Introduction

The multi-armed bandit (MAB) problem is a fundamental reinforcement learning challenge that models the exploration-exploitation trade-off in decision-making under uncertainty. This framework provides a comprehensive, production-ready implementation of various bandit algorithms and environments, enabling researchers and practitioners to study, compare, and deploy bandit-based solutions.

### Key Concepts

#### The Bandit Problem
At each time step, an agent must choose one of K possible actions (arms). Each action yields a reward drawn from some unknown probability distribution. The agent's goal is to maximize the total reward over time, which requires balancing:

1. **Exploration**: Gathering information about arms with uncertain rewards
2. **Exploitation**: Leveraging known information to maximize immediate reward

#### Regret Analysis
Regret is the primary performance metric in bandit problems, measuring the difference between the cumulative reward of the optimal strategy and the actual reward obtained by the agent.

**Cumulative Regret**:

$\qquad \rho(T) = \sum_{t=1}^T (\mu^* - \mu_{a_t})$

Where:
- $T$: Total number of time steps
- $\mu^* = \max_k\{\mu_k\}$: Expected reward of the optimal arm
- $\mu_{a_t}$: Expected reward of the arm chosen at time t
- $a_t$: Action selected at time t

**Pseudo-Regret**:

$\qquad \bar{\rho}(T) = T\mu^* - \sum_{t=1}^T \mu_{a_t}$

#### Types of Bandit Problems
1. **Stochastic Bandits**: Rewards are drawn i.i.d. from fixed distributions
2. **Adversarial Bandits**: Rewards can be chosen by an adversary
3. **Contextual Bandits**: Actions depend on additional context information
4. **Non-stationary Bandits**: Reward distributions change over time
5. **Linear Bandits**: Rewards are linear functions of action features

#### Common Performance Metrics
1. **Cumulative Regret**: Total regret over all time steps
2. **Average Regret**: Regret per time step
3. **Simple Regret**: Regret of the final recommended arm
4. **Sample Complexity**: Number of samples needed to identify a near-optimal arm
5. **Convergence Rate**: How quickly the algorithm approaches optimal performance

## Algorithms

### 1. ε-Greedy Agent

The ε-Greedy algorithm is a simple yet effective strategy that explicitly balances exploration and exploitation through a fixed exploration rate.

#### Mathematical Formulation
At each time step t:
1. With probability ε: 
   - **Explore**: Select an arm uniformly at random from all K arms
2. With probability 1-ε:
   - **Exploit**: Select the arm with the highest empirical mean reward:
   
   $a_t = \arg\max_{k} \hat{\mu}_k(t-1)$

Where:
- $\hat{\mu}_k(t-1)$ is the empirical mean reward of arm k up to time t-1
- ε is the exploration rate parameter

#### Key Parameters
- `epsilon` (float, default=0.1): Exploration rate (0 ≤ ε ≤ 1)
  - Higher ε leads to more exploration
  - Lower ε leads to more exploitation
- `environment_type` (str, default='bernoulli'): Type of environment ('bernoulli' or 'gaussian')

#### Variants
1. **Decaying ε-Greedy**: Decreases ε over time
   - Common schedule: $\epsilon_t = \min(1, \frac{cK}{d^2t})$ where c > 0 and d is the minimum suboptimality gap
2. **Adaptive ε-Greedy**: Adjusts ε based on observed rewards
3. **Annealed ε-Greedy**: Uses a decreasing exploration schedule

#### Theoretical Guarantees
- **Regret Bound**: $O(\frac{K\log T}{\Delta})$ where Δ is the minimum suboptimality gap
- **Convergence**: Almost surely converges to the optimal arm
- **Sample Complexity**: $O(\frac{K}{\Delta^2} \log \frac{1}{\delta})$ to identify the best arm with probability 1-δ

#### Implementation Details
- Uses efficient incremental updates for mean estimation
- Handles both Bernoulli and Gaussian reward distributions
- Supports parallel execution for multiple runs

#### When to Use
- Simple to implement and understand
- Works well when the number of arms is small
- Good baseline for comparison with more sophisticated algorithms
- Suitable for non-stationary environments when combined with appropriate ε-decay

#### Example Usage
```python
from agents.epsilon import EpsilonGreedyAgent

# Initialize agent with ε=0.1 for a Bernoulli bandit
agent = EpsilonGreedyAgent(epsilon=0.1, environment_type='bernoulli')

# Initialize with 5 actions
agent.init_actions(n_actions=5)

# Get an action
action = agent.get_action()
# Update with observed reward
agent.update(action=action, reward=1.0)
```

### 2. UCB (Upper Confidence Bound) Agent

The UCB (Upper Confidence Bound) algorithm is a more sophisticated approach that uses confidence intervals to balance exploration and exploitation. The key idea is to maintain an upper confidence bound for each arm's reward and select the arm with the highest UCB.

#### Mathematical Formulation
At each time step t, select the arm that maximizes:

$\text{UCB}_k(t) = \hat{\mu}_k(t) + c \sqrt{\frac{2 \ln t}{N_k(t)}}$

Where:
- $\hat{\mu}_k(t)$ is the empirical mean reward of arm k at time t
- $N_k(t)$ is the number of times arm k has been pulled up to time t
- c is a parameter controlling the exploration-exploitation trade-off (typically c=1)
- t is the current time step

#### Key Parameters
- `c` (float, default=1.0): Exploration parameter
  - Higher c leads to more exploration
  - Lower c leads to more exploitation
- `environment_type` (str, default='gaussian'): Type of environment ('bernoulli' or 'gaussian')

#### Variants
1. **UCB1**: The standard UCB algorithm with c=1
2. **UCB-Tuned**: Adapts the exploration term based on reward variance
3. **KL-UCB**: Uses Kullback-Leibler divergence for tighter bounds
4. **MOSS (Minimax Optimal Strategy in the Stochastic case)**: Optimized for finite-time performance

#### Theoretical Guarantees
- **Regret Bound**: $O(\frac{K\log T}{\Delta})$ where Δ is the minimum suboptimality gap
- **Optimality**: Order-optimal for stochastic bandits
- **Convergence**: Almost surely converges to the optimal arm

#### Implementation Details
- Efficiently maintains running statistics
- Handles both Bernoulli and Gaussian reward distributions
- Uses numerical stability techniques for small sample sizes
- Supports parallel execution for multiple runs

#### When to Use
- When you need theoretical guarantees on performance
- When the number of arms is moderate
- When the reward distributions have bounded support
- When you want to minimize cumulative regret

#### Example Usage
```python
from agents.ucb import UCBAgent

# Initialize UCB agent with c=1.0
agent = UCBAgent(c=1.0, environment_type='gaussian')

# Initialize with 10 actions
agent.init_actions(n_actions=10)

# Get an action
action = agent.get_action()
# Update with observed reward
agent.update(action=action, reward=0.75)
```

#### Advanced: UCB-Tuned
UCB-Tuned adapts the exploration term based on the estimated variance of each arm:

$\text{UCB-Tuned}_k(t) = \hat{\mu}_k(t) + \sqrt{\frac{\ln t}{N_k(t)} \min(1/4, V_k(N_k(t)))}$

Where $V_k(t)$ is an estimate of the variance of arm k at time t.

### 3. Thompson Sampling Agent

Thompson Sampling is a Bayesian algorithm that maintains a probability distribution over the possible rewards of each arm. It selects arms by sampling from these distributions and updating them based on observed rewards.

#### Mathematical Formulation

1. **Prior Distribution**:
   For each arm k, maintain a Beta(α_k, β_k) prior distribution over the reward probability θ_k
   - Initially: α_k = 1, β_k = 1 (uniform prior)

2. **Action Selection**:
   At each time step t:
   - For each arm k, sample θ_k ~ Beta(α_k, β_k)
   - Select arm a_t = argmax_k θ_k

3. **Posterior Update**:
   After observing reward r_t ∈ {0,1}:
   - If r_t = 1: α_{a_t} += 1
   - Else: β_{a_t} += 1

For Gaussian rewards with unknown mean and variance, we use a Normal-Gamma prior.

#### Key Parameters
- `environment_type` (str, default='bernoulli'): Type of environment ('bernoulli' or 'gaussian')
- `prior_alpha` (float, default=1.0): Initial alpha parameter for Beta prior
- `prior_beta` (float, default=1.0): Initial beta parameter for Beta prior

#### Variants
1. **Gaussian Thompson Sampling**: For normally distributed rewards
2. **Non-stationary Thompson Sampling**: For changing reward distributions
3. **Contextual Thompson Sampling**: For contextual bandit problems
4. **Hierarchical Thompson Sampling**: For grouped bandit problems

#### Theoretical Guarantees
- **Regret Bound**: $O(\frac{K\log T}{\Delta})$ where Δ is the minimum suboptimality gap
- **Bayesian Regret**: Minimizes Bayesian regret under certain conditions
- **Convergence**: Almost surely converges to the optimal arm

#### Implementation Details
- Efficient sampling using NumPy's random number generation
- Handles both Bernoulli and Gaussian reward distributions
- Uses numerical stability techniques for small sample sizes
- Supports parallel execution for multiple runs

#### When to Use
- When you want a simple yet effective Bayesian approach
- When the number of arms is small to moderate
- When you need good empirical performance
- When you want to incorporate prior knowledge

#### Example Usage
```python
from agents.thompson import ThompsonSamplingAgent

# Initialize Thompson Sampling agent for Bernoulli rewards
agent = ThompsonSamplingAgent(environment_type='bernoulli')

# Initialize with 3 actions
agent.init_actions(n_actions=3)

# Get an action
action = agent.get_action()
# Update with observed reward (0 or 1 for Bernoulli)
agent.update(action=action, reward=1)
```

#### Advanced: Gaussian Thompson Sampling
For Gaussian rewards with unknown mean and variance, we use a Normal-Gamma prior:

1. Prior: $\mu_k, \tau_k \sim \text{NormalGamma}(\mu_0, \lambda_0, \alpha_0, \beta_0)$
2. Sample $\tau_k \sim \text{Gamma}(\alpha_k, \beta_k)$
3. Sample $\mu_k \sim \mathcal{N}(\mu_k, (\lambda_k \tau_k)^{-1})$
4. Select arm with highest sampled $\mu_k$
5. Update posterior parameters based on observed reward

### 4. KL-UCB Agent

The KL-UCB (Kullback-Leibler Upper Confidence Bound) algorithm is an improvement over standard UCB that uses the Kullback-Leibler divergence to obtain tighter confidence bounds, particularly for Bernoulli and other exponential family distributions.

#### Mathematical Formulation

For Bernoulli rewards, at each time step t, select the arm that maximizes:

$\text{KL-UCB}_k(t) = \max\left\{ q \in [\hat{\mu}_k(t), 1] : N_k(t) \cdot d(\hat{\mu}_k(t), q) \leq \log(t) + c\log(\log(t)) \right\}$

Where:
- $\hat{\mu}_k(t)$ is the empirical mean of arm k at time t
- $N_k(t)$ is the number of times arm k has been pulled up to time t
- $d(p,q) = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$ is the binary KL divergence
- c is a parameter (typically c=3)

For Gaussian rewards with known variance $\sigma^2$:

$\text{KL-UCB-Gaussian}_k(t) = \hat{\mu}_k(t) + \sqrt{\frac{2\sigma^2(\log(t) + c\log(\log(t)))}{N_k(t)}}$

#### Key Parameters
- `c` (float, default=3.0): Exploration parameter
- `tolerance` (float, default=1e-4): Numerical tolerance for solving the KL-UCB equation
- `max_iter` (int, default=100): Maximum number of iterations for the numerical solver

#### Theoretical Guarantees
- **Regret Bound**: $O(\frac{K\log T}{\Delta})$ where Δ is the minimum suboptimality gap
- **Optimality**: Asymptotically optimal for Bernoulli bandits
- **Finite-time Performance**: Often outperforms UCB in practice, especially for small sample sizes

#### Implementation Details
- Uses binary search to solve the KL-UCB equation
- Handles both Bernoulli and Gaussian reward distributions
- Implements efficient numerical methods for stability
- Supports parallel execution for multiple runs

#### When to Use
- When rewards are binary or bounded
- When you need optimal asymptotic performance
- When the number of arms is small to moderate
- When you want better empirical performance than UCB

#### Example Usage
```python
from agents.ucb_kl import KLUCBAgent

# Initialize KL-UCB agent with c=3.0
agent = KLUCBAgent(c=3.0)

# Initialize with 5 actions
agent.init_actions(n_actions=5)
# Get an action
action = agent.get_action()
# Update with observed reward
agent.update(action=action, reward=1.0)
```

#### Advanced: Solving the KL-UCB Equation
The KL-UCB equation is solved using binary search:

1. Initialize q = 1
2. While d(μ,q) > (log(t) + c*log(log(t)))/N_k(t):
   - q = (μ + q)/2
3. Return q

This finds the maximum q such that the KL divergence constraint is satisfied.

### 5. Gaussian Thompson Sampling Agent
Variant of Thompson Sampling for normally distributed rewards.

**Mathematical Formulation**:
1. Maintain Normal-Inverse-Gamma conjugate prior for each arm
2. At each step:
   - Sample (μ, σ²) from posterior
   - Select arm with highest sampled μ
   - Update posterior parameters based on observed reward

### 6. LLM-based Agent
An experimental agent that uses large language models for action selection, combining learned priors with exploration strategies.

## Environments

### 1. Bernoulli Bandit
Each arm returns a reward of 1 with probability p and 0 otherwise, where p is specific to each arm.

### 2. Gaussian Bandit
Each arm returns a reward sampled from a normal distribution N(μ, σ²), where μ and σ are specific to each arm.

## Project Structure

```
bandit_algos/
├── agents/                    # Bandit algorithm implementations
│   ├── base_agent.py          # Base class for all agents
│   ├── epsilon.py             # ε-greedy agent
│   ├── ucb.py                 # UCB agent
│   ├── thompson.py            # Thompson Sampling agent
│   ├── ucb_kl.py              # KL-UCB agent
│   ├── gaussian_ucb.py        # Gaussian UCB agent
│   ├── gaussian_thompson_sampling.py  # Gaussian Thompson Sampling
│   ├── llm_agent.py           # LLM-based agent
│   └── llm_agentV2.py         # Enhanced LLM-based agent
│
├── environments/             # Bandit environments
│   ├── base_environment.py    # Base environment class
│   ├── bernoulli_bandit.py    # Bernoulli bandit implementation
│   └── gaussian_bandit.py     # Gaussian bandit implementation
│
├── configurations/           # Configuration files
│   ├── experiment/            # Experiment configurations
│   ├── agent/                 # Agent configurations
│   └── environment/           # Environment configurations
│
├── plots/                    # Plotting utilities
├── results/                   # Experiment results
├── outputs/                   # Output files
├── utils/                     # Utility functions
└── tests/                     # Test files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bandit-algos.git
cd bandit-algos
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

To run a basic experiment:
```bash
python main.py
```

### Configuration

Modify the configuration files in the `configurations/` directory to customize experiments:

- `configurations/experiment/experiment.yaml`: Main experiment settings
- `configurations/agent/*.yaml`: Agent-specific configurations
- `configurations/environment/*.yaml`: Environment configurations

Example experiment configuration:
```yaml
experiment:
  output_dir: output
  n_steps: 1000
  n_runs: 10
  seed: 42
seeds:
  numpy: 42
  random: 42
```

## Advanced Features

### Delayed Feedback
Support for environments with delayed rewards, simulating real-world scenarios where feedback isn't immediate.

### Non-stationary Bandits
Implementations that can handle environments where reward distributions change over time.

### Custom Environment Creation
Easily create custom bandit environments by extending the base environment class.

## Results and Visualization

After running an experiment, results will be saved in the `outputs/` directory, including:
- Regret curves
- Action selection frequencies
- Reward distributions
- Performance metrics

Visualizations are generated using matplotlib and can be customized in the `plots/` directory.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contributors
- Donald Zvada
- Maxime Heuillet
- Audrey Durand
- Kudziedevs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
