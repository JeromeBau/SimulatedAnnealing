import random
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import constants


class SimulatedAnnealing(object):
    """
    Class that applies Simulated Annealing to a continuous optimization problem of a simple function
    """
    def __init__(self):
        self.kmax=100
        self.state_evolution = []
        self.neighbor_function = 'normal'
        self.temperature_initial = 1.0
        self.temperature_terminal = 0.0001
        self.cooling_rate = 0.9

    def energy(self, s):
        """ State energery / Cost function """
        return 0.01*s**2 + 40 * np.sin(0.3*s)

    def neighbor(self, s):
        """ Return a random neighbor
        Possible neighbor functions:
        (1) Uniform distributed random neighbor
        (2) Normally distributed random neighbor
        """
        if self.neighbor_function == 'uniform':
            return s + random.uniform(-100, 100)
        elif self.neighbor_function == 'normal':
            return s + random.normal(0, 100)

    @staticmethod
    def acceptance_probability(energy_old, energy_new, T):
        """ Returns an acceptance probability
        Based on the comparison of old and new value and the according temperature
        """"
        return np.exp((energy_old-energy_new)/T)

    def anneal(self, state):
        """ Annealing process """
        self.state_evolution.append(state)
        energy_old = self.energy(state)
        T = self.temperature_initial
        while T > self.temperature_terminal:
            n = 1
            while n < 200:
                new_state = self.neighbor(state)
                energy_new = self.energy(new_state)
                if self.acceptance_probability(energy_old, energy_new, T) > random.random():
                    state = new_state
                    energy_old = energy_new
                    self.state_evolution.append(new_state)
                n += 1
            T = T*self.cooling_rate
        return state, energy_old
