# Multi-Armed Bandit Simulation

This repository contains a Python implementation of several exploration strategies for the Bernoulli multi-armed bandit problem. The project emphasizes modularity, testability, and ease of comparison between different bandit algorithms by using a clear separation of concerns.

## Overview

The multi-armed bandit (MAB) problem is a classic reinforcement learning challenge.  An agent repeatedly chooses between multiple actions (arms), each with an unknown reward probability. The primary goal is to minimize *regret*, which represents the cumulative difference between the rewards obtained and what could have been achieved by consistently selecting the optimal action.

Mathematically, the regret $\rho$ over a time horizon $T$ is defined as:

$\qquad \rho = T\theta^* - \sum_{t=1}^T r_t$

where:

*   $T$ is the total number of time steps (or trials).
*   $\theta^* = \max_k\{\theta_k\}$ is the optimal reward probability, i.e., the maximum reward probability across all arms.
*   $r_t$ is the reward received at time $t$.

In the **Bernoulli bandit** setting, each arm $k$ yields a reward of 1 (success) with probability $\theta_k$ and a reward of 0 (failure) with probability $1 - \theta_k$.  The agent does not know the true values of $\theta_k$ for each arm and must learn them through exploration and exploitation.

## Modularity and File Structure

The code is organized into distinct modules to improve readability, maintainability, and extensibility.

*   **`bernoulli_bandit.py`:** Defines the `BernoulliBandit` environment.  This class encapsulates the logic for simulating the bandit problem, including:

    *   `pull(action)`:  Simulates pulling a specific arm and returns a reward (0 or 1) based on the arm's underlying probability.
    *   `optimal_reward()`:  Returns the maximum reward probability, used for calculating regret.
    *   `reset()`: Resets the bandit's state to its initial configuration.

*   **`abstract_agent.py`:** Defines the abstract base class `AbstractAgent`. This class provides a common interface for all agent implementations.  Key methods include:

    *   `get_action()`: An abstract method that each agent must implement to choose an action.
    *   `update(action, reward)`:  Updates the agent's internal state based on the received reward.
    *   `init_actions(n_actions)`: Initializes the agent's internal state variables.

*   **`epsilon.py`, `ucb.py`, `thompson.py`:** These files implement specific bandit algorithms as concrete classes inheriting from `AbstractAgent`. They provide detailed implementations for the `get_action()` method, defining how each agent selects actions based on its particular strategy.

    ### Algorithm Details and Mathematical Formulations

    *   **Epsilon-Greedy Agent:**

        This algorithm balances exploration and exploitation.  With probability $\epsilon$, the agent explores by selecting a random arm.  Otherwise, with probability $1 - \epsilon$, the agent exploits by choosing the arm with the highest estimated reward.

        **Algorithm:**

        1.  **Initialization:** Set $\epsilon \in (0, 1)$. Initialize estimates $\hat{\theta}_k = 0$ for all arms $k$. Initialize counts $n_k = 0$ for all arms $k$.

        2.  **For each time step** $t = 1, 2, \dots, T$:

            *   Generate a random number $p \sim \text{Uniform}(0, 1)$.

            *   **If** $p < \epsilon$ (Exploration):
                *   Select an arm $a_t$ uniformly at random from the set of all arms.

            *   **Else** (Exploitation):
                *   Select the arm $a_t = \arg\max_k \hat{\theta}_k$.

            *   Observe the reward $r_t$ from arm $a_t$.

            *   Update the estimate for arm $a_t$:

                $\qquad n_{a_t} = n_{a_t} + 1$

               $\hat{\theta} = \hat{\theta} + \frac{1}{n}(r_t - \hat{\theta})$


        **Variables:**

        *   $\epsilon$: Exploration rate (probability of choosing a random action).
        *   $\hat{\theta}_k$: Estimated reward for arm $k$.
        *   $n_k$: Number of times arm $k$ has been selected.
        *   $a_t$: Action selected at time $t$.
        *   $r_t$: Reward received at time $t$.

    *   **UCB1 Agent (Upper Confidence Bound):**

        The UCB1 algorithm aims to balance exploration and exploitation by selecting arms based on both their estimated rewards and the uncertainty associated with those estimates.  It adds an exploration bonus to the estimated reward, favoring arms that haven't been tried often.

        **Algorithm:**

        1.  **Initialization:** Initialize estimates $\hat{\theta}_k = 0$ for all arms $k$. Initialize counts $n_k = 0$ for all arms $k$.

        2.  **For each time step** $t = 1, 2, \dots, T$:

            *   Calculate the UCB for each arm $k$:

                $\qquad UCB_k = \hat{\theta}_k + \sqrt{\frac{2 \ln t}{n_k}}$

            *   Select the arm $a_t = \arg\max_k UCB_k$.

            *   Observe the reward $r_t$ from arm $a_t$.

            *   Update the estimate for arm $a_t$:

                $\qquad n_{a_t} = n_{a_t} + 1$

                $\hat{\theta} = \hat{\theta} + \frac{1}{n}(r_t - \hat{\theta})$


        **Variables:**

        *   $\hat{\theta}_k$: Estimated reward for arm $k$.
        *   $n_k$: Number of times arm $k$ has been selected.
        *   $t$: Current time step.
        *   $UCB_k$: Upper Confidence Bound for arm $k$.
        *   $a_t$: Action selected at time $t$.
        *   $r_t$: Reward received at time $t$.

    *   **Thompson Sampling Agent:**

        Thompson Sampling uses a Bayesian approach to balance exploration and exploitation. It maintains a probability distribution (in this case, a Beta distribution) for each arm, representing the uncertainty about its reward probability. It samples from these distributions to choose the next arm to play.

        **Algorithm:**

        1.  **Initialization:** For each arm $k$, initialize a Beta distribution with parameters $\alpha_k = 1$ and $\beta_k = 1$ (representing no prior knowledge).  This gives each arm a uniform prior.

        2.  **For each time step** $t = 1, 2, \dots, T$:

            *   For each arm $k$, sample a reward probability $\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$.

            *   Select the arm $a_t = \arg\max_k \theta_k$.

            *   Observe the reward $r_t$ from arm $a_t$.

            *   Update the parameters of the Beta distribution for arm $a_t$:

                *   **If** $r_t = 1$ (Success):
                    $\qquad \alpha_{a_t} = \alpha_{a_t} + 1$
                *   **Else** (Failure):
                    $\qquad \beta_{a_t} = \beta_{a_t} + 1$

        **Variables:**

        *   $\alpha_k$: Parameter of the Beta distribution for arm $k$ representing the number of successes.
        *   $\beta_k$: Parameter of the Beta distribution for arm $k$ representing the number of failures.
        *   $\theta_k$: Sampled reward probability for arm $k$.
        *   $a_t$: Action selected at time $t$.
        *   $r_t$: Reward received at time $t$.

*   **`plots.py`:** Contains the `plot_regret()` function, which generates a plot of the cumulative regret for each agent over time.  This allows for visual comparison of the performance of different algorithms.

*   **`main.py`:** The main script that orchestrates the simulation.  It:
    1.  Initializes the `BernoulliBandit` environment.
    2.  Creates instances of the agent classes.
    3.  Runs multiple trials of the bandit problem, allowing each agent to interact with the environment and learn.
    4.  Calculates the cumulative regret for each agent.
    5.  Calls `plot_regret()` to visualize the results.

*   **`tests/test_agents.py`:** Contains unit tests to verify the basic functionality of the agent implementations.  These tests ensure that the agents are correctly selecting actions, updating their internal state, and handling edge cases.

## Configuration

Configuration files are now stored in the `configurations/` folder, using YAML format for clarity and flexibility. The structure is as follows:

- `configurations/config.yaml`: Main configuration file with defaults.
- `configurations/agent/`: Agent configuration files (e.g., `epsilon_greedy.yaml`, `ucb.yaml`, etc.).
- `configurations/environment/`: Environment configuration files (e.g., `bernoulli_env.yaml`, `gaussian_env.yaml`).
- `configurations/experiment/`: Experiment configuration files (e.g., `experiment.yaml`).

You can modify these YAML files to change the experiment setup, agent parameters, or environment details.

## Dependencies

* Python 3.12.4
* numpy
* pandas
* matplotlib
* omegaconf

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the Project

Simply run:

```bash
python main.py
```

The project will use the YAML configuration files in `configurations/` to set up and run the experiments.
