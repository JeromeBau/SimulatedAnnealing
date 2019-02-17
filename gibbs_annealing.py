import random
import itertools
import logging
import numpy as np
from numpy import random
import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from sklearn import datasets
import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class SimulatedAnnealingGibbs(object):
    """" Applies Simulated Annealing to selection of variables
    """
    def __init__(self):
        self.temperature_initial = 1
        self.temperature_terminal = 0.0001
        self.cooling_rate = 0.9
        self.X = None
        self.y = None
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

    def generate_data(self, features, info, noise, seed):
        """ Generate a data set 'from scratch' for a regression

        :param features: number of features in total
        :param info: number of features that are to be kept (informative)
        :param noise: degree of noise
        """
        data, y, coef = datasets.make_regression(n_samples=1000, n_features=features, n_informative=info, noise=noise, coef=True,
                                        random_state=seed)
        self.X = pd.DataFrame(data)
        self.y = pd.Series(y)

    def initiate_variables(self):
        """Generate at first a 'subset' that equals the vector [1,...1] of length p"""
        self.variable_list = pd.Series(self.X.columns.tolist())
        self.subset = [1 for i in self.X.columns]

    def translate_to_variables(self, vec):
        """ Given a boolean vector [1,0,1,1,..] chose variables accordingly"""
        return self.variable_list[pd.Series(vec) == 1].tolist()

    def energy(self, measure='bic', on='subset'):
        """ Energy function of the annealing progress
        Calculate an OLS model for either 'subset' (current state) or 'neighbor' (alternative state).
        Measures the costs of the current model using either 'aic' or 'bic'
        """
        if on == 'subset':
            if sum(self.subset) == 0:  # Case: No variables selected
                return 99999999
            model = sm.OLS(self.y, self.X[self.translate_to_variables(self.subset)])
        elif on == 'neighbor':
            if sum(self.neighbor) == 0:
                return 99999999
            model = sm.OLS(self.y, self.X[self.translate_to_variables(self.neighbor)])
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

    def generate_gibbs_neighbor(self, n):
        """ Switch the boolean state for the nth component"""
        self.neighbor = self.subset.copy()
        self.neighbor[n] = 1-self.neighbor[n]  # replace 1 with 0 and 0 with 1

    def anneal(self):
        """ Simulated Annealing
        Work through the classical process of
        """
        self.state_evolution.append(self.subset)  # state_evolution only serves to follow up on the states chosen
        T = self.temperature_initial
        while T > self.temperature_terminal:
            for n in range(len(self.subset)):
                self.generate_gibbs_neighbor(n)
                energy_old = self.energy(on='subset')
                energy_new = self.energy(on='neighbor')
                if self.acceptance_probability(energy_old, energy_new, T) > random.random():
                    self.subset = self.neighbor  # the alternative state was accepted
                    self.state_evolution.append(self.subset)
            T *= self.cooling_rate
        return self.subset

    def optimize_on_full_search_space(self):
        """ Go through the whole search space and determine the true optimum"""
        logging.debug('This might take a while. You might wanna go grab a coffee. Or two.')
        all_possible_subsets = []
        for m in range(1, len(self.variable_list)+1):
            for i in itertools.combinations(self.variable_list, m):
                all_possible_subsets.append(i)
        check = [list(i) for i in all_possible_subsets]
        p = pd.DataFrame(pd.Series(check))
        p['score'] = p[0]

        def en(x):
            model = sm.OLS(self.y, self.X[x])
            results = model.fit()
            return results.bic

        p['score'] = p['score'].apply(en)
        return p

    def _test(self, features, info, noise):
        """Run algorithm with specific parameters and compare with actual optimum"""
        self.generate_data(features=features, info=info, noise=noise)
        self.initiate_variables()
        t0 = time.time()
        best_guess = self.anneal()
        t1 = time.time()
        p = v.optimize_on_full_search_space()
        t2 = time.time()
        best = p[p['score'] == p['score'].min()][0].tolist()[0]
        return (best, t2-t1, self.translate_to_variables(best_guess), t1-t0)

    def _test_full_search(self, features, info, noise=1):
        """Rund algorithm with specific parameters and compare with actual optimum"""
        self.generate_data(features=features, info=info, noise=noise)
        self.initiate_variables()
        t0 = time.time()
        p = v.optimize_on_full_search_space()
        t1 = time.time()
        best = p[p['score'] == p['score'].min()][0].tolist()[0]
        return t1-t0

    def _test_all(self, n1=5, n2=3):
        """ Run algorithm for different inputs and compare results to the actual optimum"""
        results = []
        f = 0
        for i in range(n1):
            f += 5
            for t in range(n2):
                param1 = f
                param2 = random.randint(1,f)
                param3 = random.randint(1, 5)
                res = self._test(features=param1, info=param2, noise=param3)
                logging.info('Results for features = ' + str(f) + ', info = ' + str(param2)
                             + ', and noise = ' + str(param3))
                logging.info('Actual optimum: ' + str(res[0]) + ', best guess: ' + str(res[1]))
                results.append([param1, param2, param3, res[0], res[1], res[2], res[3], res[0] == res[2]])
        return pd.DataFrame(results)







