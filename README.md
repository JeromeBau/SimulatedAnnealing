# <p align='center'>Simulated Annealing</p>

This repository contains several codes concerning the implementation of Simulated Annealing in Python, particularly an implementation of Simulated Annealing using a Gibbs kernel, which allows for an application of a *Traveling Salesman* type problem and also a Variable Selection Problem for a linear regression.

## Content
* Basic Idea
* Application: Continous function
* Modification for discrete search space
* Implementation of a Gibbs kernel
* Application: Variable Selection 
* Application: Traveling Salesman Problem


## Simulated Annealing - The basic idea

Simulated Annealing is a method that borrows ideas from statistical physics to optimize on a cost function on a a large search space. The idea comes from the cooling process of metal, where the cooling is carried out in such a way that at each temperature interval the molecules can align in a way that leads to a near perfect result.
The concept can be easily adapted to fit either a discrete case or a continous function.

At each temperature, the solid needs to reach its *thermal equilibrium*, which is expressed by a probability distribution, the Boltzmann distribution.

Important distinction to many other algorithms that attempt to determine an optimum: The algorithm will also go into the "wrong" direction with a certain probability. This probability decreases with the decrease of the temperature. 

A threshold depending on the temperature will stop the algorithm. 

## Application: Finding the global optimum of a continuous function

To find the optimum of a continous function, two things are adapted with respect to the general simulated annealing algorithm:

(1) Energy function

The energy function corresponds straight to the continuous function of which we want to find he optimum.

(2) Neighbor function

A neighbor can be selected through different methods. Here I decide to choose a neighbor by selecting a random point in proximity of the current point. This can be done using a uniform distribution (hence giving each neighbor the same chance) or a normal distribution (giving closer neighbors better chances). The choice of variance parameter is clearly of high significance: The higher the variance, the larger the "jumps" from one neighbor to another. 



<p align="center">
<img src="https://raw.githubusercontent.com/JeromeBau/SimulatedAnnealing/master/simulated_annealing_example.gif" alt='Simple example for a simulated annealing algorithm'/>
</p>


## Modification for discrete search space

In progress

## Implementation of a Gibbs kernel

In progress

## Applying the algorithm to Variable Selectio in a linear Regression

In progress

## Applying the algorithm to the Travelin Salesman Problem

In progress
