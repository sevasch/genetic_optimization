import numpy as np
import matplotlib.pyplot as plt

class GeneticOptimizer():
    def __init__(self,
                 member_class,
                 p_elitism: float,
                 p_crossover: float,
                 p_mutate: float,
                 *member_args,
                 crossover_coeff: float=0.2,
                 reverse_fitness=False):
        assert p_elitism + p_crossover + p_mutate <= 1
        self.member_class = member_class
        self.member_args = member_args
        self.p_elitism = p_elitism
        self.p_crossover = p_crossover
        self.p_mutate = p_mutate
        self.crossover_coeff = crossover_coeff  # low --> diverse parents
        self.reverse_fitness = reverse_fitness
        self._population = []
        self._population_history = []
        self._fitness_history = []


    def _create_intial_population(self,
                                  population_size):
        return [self.member_class.create_random(*self.member_args) for _ in range(population_size)]


    def _step_population(self,
                         population_size) -> list:

        # rank according to fitness
        self._population.sort(key=lambda member: member.get_fitness(), reverse=self.reverse_fitness)

        # select best members for elitism
        new_population = [self._population[i].let_survive() for i in range(int(self.p_elitism * population_size))]

        # apply crossover
        for _ in range(int(self.p_crossover * population_size)):
            member_indices = np.zeros(2)
            while (np.all(member_indices == member_indices[0]) or (max(member_indices) >= len(self._population))):
                member_indices = np.random.geometric(p=self.crossover_coeff, size=2)
            new_population.append(self._population[member_indices[0]].crossover(self._population[member_indices[1]]))

        # apply mutations
        for member in new_population:
            if np.random.random() < self.p_mutate:
                new_population.append(member.mutate())

        # fill up with random members
        for _ in range(population_size - len(new_population)):
            new_population.append(self.member_class.create_random(*self.member_args))

        self._population = new_population


    def run_evolution(self,
                      n_generations,
                      population_size,
                      random_seed=None):
        # np.random.seed(random_seed)  #TODO: not working

        self._population = self._create_intial_population(population_size)
        self._population_history.append(self._population)
        self._fitness_history.append(self.get_best_member().get_fitness())

        print('generation ' + str(0).zfill(4) + ', best fitness value: ' + str(self.get_best_member().get_fitness()))

        for generation_no in range(n_generations):
            self._step_population(population_size)
            self._population_history.append(self._population)
            self._fitness_history.append(self.get_best_member().get_fitness())
            print('generation ' + str(generation_no + 1).zfill(4) + ', best fitness value: ' + str(self.get_best_member().get_fitness()))

    #TODO: to apply stepping of probabilities, create new run method and step values in between

    def get_current_population(self):
        return sorted(self._population, key=lambda member: member.get_fitness(), reverse=self.reverse_fitness)

    def get_best_member(self):
        return sorted(self._population, key=lambda member: member.get_fitness(), reverse=self.reverse_fitness)[0]

    def get_history(self):
        return (self._population_history, self._fitness_history)

    def plot_fitness_history(self):
        plt.plot(self._fitness_history)
        plt.xlabel('generation'), plt.ylabel('best fitness value')
        plt.xlim([0, len(self._fitness_history)])
        plt.grid()
        plt.show()