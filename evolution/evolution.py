import numpy as np
from torch import from_numpy

class GenericEvolution():

    def __init__(self, genes, pop_size, fitness_func, lr=0.1, generations=100,
                    base_indv=None):
        self.genes = genes
        self.pop_size = pop_size
        self.calc_fitness = fitness_func
        self.generations = generations
        self.lr = lr
        self.base_indv = base_indv

        self.population = None
        self.max_fitness = -1
        self.best_indv = None

    def init_population(self):
        if self.base_indv is None:
            self.population = np.random.rand(self.pop_size, self.genes)
        else:
            if self.genes != len(self.base_indv):
                print("Warning: Using # genes of based individual.")
            self.population = np.array([self.base_indv] * self.pop_size)

    def evolve(self):
        for gen in range(self.generations):
            # mutating population
            mutations = np.random.normal(
                                        loc=0, scale=1,
                                        size=(self.pop_size, self.genes)
                                        )

            mutated_pop = self.population + mutations * self.lr

            # calculating fitness scores for resulting individuals, saving best
            fitness_scores = [self.calc_fitness(indv) for indv in mutated_pop]
            self.update_best(fitness_scores, mutated_population)

            # sampling from mutation population based on fitness
            fitness_score_prob = fitness_scores / sum(fitness_scores)
            new_population_idx = np.random.choice(
                                            len(mutated_pop),
                                            p=fitness_score_prob,
                                            size=self.pop_size
                                            )

            self.population = mutated_pop[new_population_idx]

    def update_best(self, fitness_scores, mutated_population):
        max_idx = fitness_scores.index(max(fitness_scores))
        if fitness_scores[max_idx] > self.max_fitness:
            self.best_indv = mutated_population[max_idx]
            self.max_fitness = fitness_scores[max_idx]


class FitnessBasedSamplingEvolution():
    def __init__(self, model, pop_size, fitness_func, noise_mean=0,
                    noise_std=0.1, generations=10):
        self.model = model
        self.pop_size = pop_size
        self.calc_fitness = fitness_func
        self.generations = generations
        self.noise_std = noise_std
        self.noise_mean = noise_mean

        self.population = None
        self.max_fitness = -10000
        self.best_indv = None

        self.base_indv = np.array([p.data.numpy() for p in model.parameters()])
        self.p_shapes = np.array([p.shape for p in self.base_indv])

    def init_population(self):
        return np.array([self.base_indv] * self.pop_size)

    def generate_mutations(self):
        mutations = []
        for indv in range(self.pop_size):
            m = []
            for shape in self.p_shapes:
                m.append(np.random.normal(
                            loc=self.noise_mean, scale=self.noise_std,
                            size=shape
                            ))
            mutations.append(np.array(m))
        return np.array(mutations)

    def evolve(self):
        if self.population is None:
            self.population = self.init_population()

        for gen in range(self.generations):
            # mutating population
            mutations = self.generate_mutations()
            mutated_pop = self.population + mutations

            # calculating fitness scores for resulting individuals, saving best
            fitness_scores = []
            for indv in mutated_pop:
                self.update_model_params(indv)
                fitness_scores.append(self.calc_fitness(self.model))

            #print(fitness_scores)
            self.update_best(fitness_scores, mutated_pop)

            # sampling from mutation population based on fitness
            fitness_score_prob = np.array(fitness_scores) / sum(fitness_scores)
            for i, (score, prob) in enumerate(zip(fitness_scores, fitness_score_prob)):
                pass #print("{} | {:.5f} {:.3f}".format(i, score, prob))

            new_population_idx = np.random.choice(
                                            len(mutated_pop),
                                            p=fitness_score_prob,
                                            size=self.pop_size
                                            )
            print(new_population_idx)
            self.population = mutated_pop[new_population_idx]
            print(gen, self.max_fitness)
            print("-----------------")

        self.update_model_params(self.best_indv)

    def update_best(self, fitness_scores, mutated_population):
        max_idx = fitness_scores.index(max(fitness_scores))
        if fitness_scores[max_idx] > self.max_fitness:
            self.best_indv = mutated_population[max_idx]
            self.max_fitness = fitness_scores[max_idx]

    def update_model_params(self, indv):
        for i, p in enumerate(self.model.parameters()):
            p.data = from_numpy(indv[i]).float()


class WeightMutationEvolution():
    def __init__(self, model, pop_size, fitness_func, noise_mean=0,
                    noise_std=0.01, generations=10, elitism=0.4):
        self.model = model
        self.pop_size = pop_size
        self.calc_fitness = fitness_func
        self.generations = generations
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.elitism_num = int(pop_size * elitism)
        self.sample_num = self.pop_size - self.elitism_num

        self.fitness_history = []

        self.max_fitness = -10000
        self.best_indv = None

        self.base_indv = np.array([p.data.numpy() for p in model.parameters()])
        self.p_shapes = np.array([p.shape for p in self.base_indv])

        self.population = np.array([self.base_indv] * self.pop_size)

    def evolve(self):
        for i in range(self.generations):
            fitness_scores = np.array([self.calc_fitness(model)
                                for model in self.generate_models()])

            print("Max Fitness Generation #{}: {}".format(i+1, max(fitness_scores)))
            self.fitness_history.append(max(fitness_scores))

            elitism_idx = fitness_scores.argsort()[-self.elitism_num:]


            fitness_prob = fitness_scores / sum(fitness_scores)
            sample_idx = np.random.choice(
                                        len(self.population),
                                        p=fitness_prob,
                                        size=self.sample_num
                                        )

            new_pop_idx = np.concatenate((elitism_idx, sample_idx))
            new_pop_base = self.population[new_pop_idx]
            self.population = self.mutate(new_pop_base)

        print(self.fitness_history)


    def generate_models(self):
        for indv in self.population:
            for i, p in enumerate(self.model.parameters()):
                p.data = from_numpy(indv[i]).float()
            yield self.model

    def mutate(self, pop):
        mutations = []
        for indv in range(self.pop_size):
            m = []
            for shape in self.p_shapes:
                m.append(np.random.normal(
                            loc=self.noise_mean, scale=self.noise_std,
                            size=shape
                            ))
            mutations.append(np.array(m))
        return pop + np.array(mutations)
