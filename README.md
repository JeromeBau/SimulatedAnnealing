# <p align='center'>Simulated Annealing</p>
## Simulated Annealing - The basic idea

Simulated Annealing is an exciting method that borrows ideas from statistical physics to approach optimization of large scale combinatorial problems. The idea comes from the cooling process of metal, where, the cooling is in such a way that at each temperature interval, the molecules can align.
- can be used for discrete and continuous optimization

At each temperature $T_i$, the solid needs to reach its _thermal equilibrium_, which is expressed by a probability distribution, the Boltzmann distribution.
The probability of being in a particular state is given by:
$$p(x) \prop \exp(-f(x) / T)$$
where f(x) is the energy at the time and T is the temperature. 

- the algorithm allows to move away a local minimum (to a point that temporarily is less optimal). The probability of this move decreases with the temperature

- if cooling happens sufficiently slow the global optimal will be found
$$T_k = T_0 C^k $$
where C is the cooling rate, often time $\approx$ 0.8
if cooling is too fast one gets stuck in a local min
<p align="center">
<img src="https://raw.githubusercontent.com/JeromeBau/SimulatedAnnealing/master/simulated_annealing_example.gif" alt='Simple example for a simulated annealing algorithm'/>
</p>

