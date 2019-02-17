import random as rnd
import numpy as np
from numpy import random

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# import matplotlib.animation as animation
from scipy import constants


class SimulatedAnnealing(object):
    """
    Class that applies Simulated Annealing to a continuous optimization problem of a simple function
    """
    def __init__(self):
        self.kmax=100
        self.state_evolution = []
        self.temperature_initial = 1.0
        self.temperature_terminal = 0.0001
        self.cooling_rate = 0.9

    @staticmethod
    def energy(s):
        """ State energery / Cost function """
        return 0.01*s**2 + 40 * np.sin(0.333333*s)

    @staticmethod
    def neighbor(s, method='normal'):
        """ Return a random neighbor
        Possible neighbor functions:
        (1) Uniform distributed random neighbor
        (2) Normally distributed random neighbor
        """
        assert type(s) == float or type(s) == int, 's was not of the correct type - int or float expected'
        if method == 'uniform':
            return s + random.uniform(-100, 100)
        elif method == 'normal':
            return s + random.normal(0, 100)

    @staticmethod
    def acceptance_probability(energy_old, energy_new, t):
        """ Returns an acceptance probability
        Based on the comparison of old and new value and the according temperature
        """
        return np.exp((energy_old-energy_new)/t)

    def anneal(self, state):
        """ Annealing process """
        self.state_evolution.append(state)
        energy_old = self.energy(state)
        t = self.temperature_initial
        while t > self.temperature_terminal:
            n = 1
            while n < 200:
                new_state = self.neighbor(state)
                energy_new = self.energy(new_state)
                if self.acceptance_probability(energy_old, energy_new, t) > random.random():
                    state = new_state
                    energy_old = energy_new
                    self.state_evolution.append(new_state)
                n += 1
            t *= self.cooling_rate
        return state, energy_old
