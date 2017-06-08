# <p align='center'>Simulated Annealing</p>

This repository contains several codes concerning the implementation of Simulated Annealing in Python, particularly an implementation of Simulated Annealing using a Gibbs kernel, which allows for an application of a *Traveling Salesman* type problem and also a Variable Selection Problem for a linear regression.

## Content
* Basic Idea
* Application: Continous function
* Discrete Search Space: Variable Selection in a Linear Regression
* Implementation of a Gibbs kernel
* Application


## Simulated Annealing - The basic idea

Simulated Annealing is a method that borrows ideas from statistical physics to optimize on a cost function on a a large search space. The idea comes from the cooling process of metal, where the cooling is carried out in such a way that at each temperature interval the molecules can align in a way that leads to a near perfect result.
The concept can be easily adapted to fit either a discrete case or a continous function.

At each temperature, the solid needs to reach its *thermal equilibrium*, which is expressed by a probability distribution, the Boltzmann distribution.

Important distinction to many other algorithms that attempt to determine an optimum: The algorithm will also go into the "wrong" direction with a certain probability. This probability decreases with the decrease of the temperature. 

A threshold depending on the temperature will stop the algorithm. 

## Application: Finding the global optimum of a continuous function
<p align="center">
<img src="https://raw.githubusercontent.com/JeromeBau/SimulatedAnnealing/master/simulated_annealing_example.gif" alt='Simple example for a simulated annealing algorithm'/>
</p>


To find the optimum of a continous function, two things are adapted with respect to the general simulated annealing algorithm:

(1) Energy function <br>
The energy function corresponds straight to the continuous function of which we want to find the optimum.

``` Python
def energy(s):
    return 0.01*s**2 + 40 * np.sin(0.3*s)
```

(2) Neighbor function <br>
A neighbor can be selected through different methods. Here I decide to choose a neighbor by selecting a random point in proximity of the current point. This can be done using a uniform distribution (hence giving each neighbor the same chance) or a normal distribution (giving closer neighbors better chances). The choice of variance parameter is clearly of high significance: The higher the variance, the larger the "jumps" from one neighbor to another. 

``` Python
def neighbor(s, method='normal'):
    if method == 'uniform':
        return s + random.uniform(-50, 50)
    elif method == 'normal':
        return s + random.normal(0, 50)
```

In these code snippets *s* is the current state - in our case a integer.


## Discrete Search Space: Variable Selection in a Linear Regression

Going from a continuous function to a discrete search space we need to modify the energy function and what a neighbor looks like, which are both essentially determined by the problem to be solved. 

In the case of variable selection for a Linear Regression we consider the question which variables from a list of variables should be used, knowing that including unnecessary variables will decrease the quality of our model. The decision is hence binary and we can view each point in the search space as a *d* dimensional vector, *d* being the number of variables in total. Each element will be either 1 (i.e. variable is chosen) or 0 (i.e. variable is not chosen).

We can punish unnecessary variables using for example the two measures AIC (Akaike information criterion) or BIC (Bayesian information criterion), which are already implemented in the python package statsmodels. 

The energy function will thus be something like this:

``` Python
model = sm.OLS(y, X[variables_selected])
results = model.fit()
return results.bic
```


## Implementation of a Gibbs kernel

The use of a Gibbs kernel only concerns the neighbor function and how this function is called. The idea remains the same as in *Gibbs sampling*: <br>
At each step we want to improve for each element of the vector the value that this element takes keeping everything else unchanged.

``` Python
def generate_gibbs_neighbor(self, n):
    self.neighbor = self.subset.copy()
    self.neighbor[n] = 1-self.neighbor[n]
```

If the state of the variable is 1 (i.e. variable is chosen) we replace it with a 0 (i.e. variable is not chosen). 

## Application

``` Python
v = SimulatedAnnealingGibbs()
v.generate_data(20, 5, 1, 2017)
v.initiate_variables()
result_binary = v.anneal()
```

``` Python
result_variable = v.translate_to_variables(result_binary)
```

This results in: [0, 1, 5, 12, 16]

