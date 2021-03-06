"""
Reference: https://github.com/xkiwilabs/Multi-Agent-DDPG-using-PTtorch-and-ML-Agents/blob/master/OUNoise.py
"""

import copy
import random
import numpy as np

from config import *

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=MU, theta=THETA, sigma=SIGMA, sigma_min=SIGMA_MIN, sigma_decay=SIGMA_DECAY):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Resduce  sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state