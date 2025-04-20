# bandits/__init__.py
# This file makes the bandits directory a package.  It can be empty,
# or it can contain code to be executed when the package is imported.

# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# Importing the necessary files
# from agents import *
# This file can also be empty, or it can contain code to be executed when the package is imported.

from agents.epsilon import EpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.thompson import ThompsonSamplingAgent
from agents.llm_agent import LLMAgent
from environments.bernoulli_bandit import BernoulliBandit
from environments.gaussian_bandit import GaussianBandit

__all__ = [
    'EpsilonGreedyAgent',
    'UCBAgent',
    'ThompsonSamplingAgent',
    'LLMAgent',
    'BernoulliBandit',
    'GaussianBandit'
]
