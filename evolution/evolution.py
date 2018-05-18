import numpy as np

class EvolutionGeneric():
    
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
    
