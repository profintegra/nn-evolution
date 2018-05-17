# geNNetic
Training neural networks using genetic algorithms.

For now, there's just one basic implementation of a genetic algorithm.

## Basic Genetic Algorithms
* Every individual in the current population gets mutated using noise drawn from gaussian distribution
* Fitness for every mutated individual is generated
* Fitness scores are normalized to a probability distribution
* New population is sampled from the mutated population based on the probability distribution (with replacement)

### Todo
* Elitism
* Breeding?

## Neural Network Training
* Literally just evolving the neural network parameters
* Fitness function could be inverse loss or accuracy type stuff
* Combinations of evolution + gradient-based training?
