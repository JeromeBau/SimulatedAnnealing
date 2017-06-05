import random
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
<<<<<<< HEAD


class SimulatedAnnealingSimple(object):
    """
    Class that applies Simulated Annealing to combinatoral optimization and visualisation on simple functions
=======
from scipy import constants


class SimulatedAnnealing(object):
    """
    Class that applies Simulated Annealing to a continuous optimization problem of a simple function
>>>>>>> 7a3f21adf5c08d0b40fe6a35d4c7b43fd69277d6
    """
    def __init__(self):
        self.kmax=100
        self.state_evolution = []
        self.neighbor_function = 'normal'
        self.temperature_initial = 1.0
        self.temperature_terminal = 0.0001
        self.cooling_rate = 0.9

    def energy(self, s):
<<<<<<< HEAD
        return 0.01*s**2 + 40 * np.sin(0.3*s)  # min at -5.20706

    def neighbor(self, s):
        if self.neighbor_function == 'neighbor':
=======
        """ State energery / Cost function """
        return 0.01*s**2 + 40 * np.sin(0.3*s)

    def neighbor(self, s):
        """ Return a random neighbor
        Possible neighbor functions:
        (1) Uniform distributed random neighbor
        (2) Normally distributed random neighbor
        """
        if self.neighbor_function == 'uniform':
>>>>>>> 7a3f21adf5c08d0b40fe6a35d4c7b43fd69277d6
            return s + random.uniform(-100, 100)
        elif self.neighbor_function == 'normal':
            return s + random.normal(0, 100)

    @staticmethod
    def acceptance_probability(energy_old, energy_new, T):
<<<<<<< HEAD
        return np.exp((energy_old-energy_new)/T)

    def anneal(self, state):
=======
        """ Returns an acceptance probability
        Based on the comparison of old and new value and the according temperature
        """"
        return np.exp((energy_old-energy_new)/T)

    def anneal(self, state):
        """ Annealing process """
>>>>>>> 7a3f21adf5c08d0b40fe6a35d4c7b43fd69277d6
        self.state_evolution.append(state)
        energy_old = self.energy(state)
        T = self.temperature_initial
        while T > self.temperature_terminal:
            n = 1
<<<<<<< HEAD
            while n <= 200:
=======
            while n < 200:
>>>>>>> 7a3f21adf5c08d0b40fe6a35d4c7b43fd69277d6
                new_state = self.neighbor(state)
                energy_new = self.energy(new_state)
                if self.acceptance_probability(energy_old, energy_new, T) > random.random():
                    state = new_state
                    energy_old = energy_new
                    self.state_evolution.append(new_state)
                n += 1
            T = T*self.cooling_rate
        return state, energy_old
<<<<<<< HEAD

    def visualize_annealing(self, show=True, save=True):
        assert len(self.state_evolution) > 0, 'Annealing has to be applied first.'
        X_MIN = -100
        X_MAX = 100
        Y_MIN = -50
        Y_MAX = 150

        def update_line(num, line):
            i = self.state_evolution[num]
            line.set_data( [i, i], [Y_MIN, Y_MAX])
            return line,

        fig = plt.figure()

        x = np.arange(X_MIN, X_MAX, 0.1)
        y = self.energy(x)

        plt.scatter(x, y)

        l , v = plt.plot(-6, -1, 6, 1, linewidth=2, color='red')

        plt.xlim(X_MIN, X_MAX)
        plt.ylim(Y_MIN, Y_MAX)
        plt.xlabel('x')
        plt.ylabel('y = 0.01 x^2 + 40 sin(0.3*x)')
        plt.title('Simulated Annealing')

        line_anim = animation.FuncAnimation(fig, update_line, len(self.state_evolution),
                                            fargs=(l, ), interval=100,
                                            blit=True, repeat=False)

        if save:
            line_anim.save('/home/jjb/Desktop/anim' +  str(random.randint(10000)) + '.gif', writer='imagemagick', fps=5)
        if show:
            plt.show()
        return line_anim


s = SimulatedAnnealingSimple()
# s.anneal(-70)
# s.visualize_annealing(show=False)
=======
>>>>>>> 7a3f21adf5c08d0b40fe6a35d4c7b43fd69277d6
