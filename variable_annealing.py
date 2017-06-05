import random
import itertools
import logging
import numpy as np
from numpy import random
import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class VariableAnnealing(object):
    """" Applies Simulated Annealing to selection of variables
    """
    def __init__(self):
        self.temperature_initial = 1
        self.temperature_terminal = 0.0001
        self.cooling_rate = 0.9
        self.X = None
        self.y = None
        self.subset_size = None
        self.variable_list = []
        self.subset = None
        self.subset_energy = None
        self.neighbor = None
        self.neighbor_energy = None
        self.state_evolution = []

    def load_example_data(self):
        """ Load sampele data from sklearn"""
        d = datasets.load_iris()
        self.X = pd.DataFrame(d.data)
        self.y = pd.Series(d.target)
        self.variable_list = self.X.columns.tolist()

    def choose_random_subset(self):
        """Choose from p predictor variable indices a random subset of length subset_size"""
        self.subset = np.random.choice(self.variable_list, self.subset_size, replace=False)
        logging.debug('Starting subset: ' + str(self.subset))

    def generate_neighbor(self):
        """
        Choose a random element of the subset and replace it
        with a random element that does not occur in the current subset
        """
        s = self.subset.copy()
        i = random.randint(0, self.subset_size)
        remaining = list(set(self.variable_list)-set(s))
        repl = np.random.choice(remaining, 1)[0]
        s[i] = repl
        self.neighbor = s

    def energy(self, measure='bic', on='subset'):
        """ Energy function of the annealing progress
        Calculate an OLS model for either 'subset' (current state) or 'neighbor' (alternative state).
        Measures the costs of the current model using either 'aic' or 'bic'
        """
        if on == 'subset':
            model = sm.OLS(self.y, self.X[self.subset])
        elif on == 'neighbor':
            model = sm.OLS(self.y, self.X[self.neighbor])
        else:
            raise KeyError("Only 'subset' and 'neighbor' are valid choices.")
        results = model.fit()
        if measure == 'aic':
            return results.aic
        elif measure == 'bic':
            return results.bic
        else:
            raise KeyError("Only 'aic' and 'bic' are valid measures.")

    @staticmethod
    def acceptance_probability(energy_old, energy_new, T):
        """ Compare energy states with respect to current temperature"""
        return np.exp((energy_old-energy_new)/T)

    def anneal(self):
        """ Simulated Annealing
        Work through the classical process of
        """
        self.state_evolution.append(self.subset)  # state_evolution only serves to follow up on the states chosen
        T = self.temperature_initial
        while T > self.temperature_terminal:
            n = 1
            while n <= 100:
                self.generate_neighbor()
                if self.acceptance_probability(self.energy(on='subset'), self.energy(on='neighbor'), T) > random.random():
                    self.subset = self.neighbor  # the alternative state was accepted
                    self.state_evolution.append(self.subset)
                n += 1
            T *= self.cooling_rate
            logging.debug('Current best subset: ' + str(self.subset))
        return self.subset

    def optimize_on_full_search_space(self):
        """ Go through the whole search space and determine the true optimum"""
        logging.warning('This might take a while. You might wanna go grab a coffee. Or two.')
        # TODO: Just a draft... A horrible draft....
        all_possible_subsets = []
        for i in itertools.permutations(self.variable_list):
            if set(list(i)[:self.subset_size])  not in all_possible_subsets:
                all_possible_subsets.append(set(list(i)[:self.subset_size]))
        check = [list(i) for i in all_possible_subsets]
        p = pd.DataFrame(pd.Series(check))
        p['score'] = p[0]

        def en(x):
            model = sm.OLS(self.y, self.X[x])
            results = model.fit()
            return results.aic

        p['score'] = p['score'].apply(en)
        return p


v = VariableAnnealing()
v.load_example_data()
v.subset_size = 2
v.choose_random_subset()
v.generate_neighbor()
v.anneal()

# Full search space
#         0       score
# 0  [0, 1]  187.906517
# 1  [0, 2]    9.671592
# 2  [0, 3]    0.779504
# 3  [1, 2]   28.298093
# 4  [1, 3]   -9.167228
# 5  [2, 3]   31.412611



