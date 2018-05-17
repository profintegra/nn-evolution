# evolution
Basic implementation of a genetic algorithm.

## What's happening
* Every individual in the current population gets mutated using noise drawn from gaussian distribution
* Fitness for every mutated individual is generated
* Fitness scores are normalized to a probability distribution
* New population is sampled from the mutated population based on the probability distribution

## Todo
* Elitism
* Breeding?
