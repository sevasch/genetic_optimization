import numpy as np
import matplotlib.pyplot as plt

class GeneticOptimizer():
    def __init__(self,
                 create_member_fun,
                 mutate_fun,
                 crossover_fun,
                 evaluation_fun,
                 p_elitism: float,
                 p_crossover: float,
                 p_mutate: float,
                 crossover_coeff: float=0.2,
                 high_is_good=False):
        assert p_elitism + p_crossover + p_mutate <= 1
        self.create_member_fun = create_member_fun
        self.mutate_fun = mutate_fun
        self.crossover_fun = crossover_fun
        self.evaluation_fun = evaluation_fun
        self.p_elitism = p_elitism
        self.p_crossover = p_crossover
        self.p_mutate = p_mutate
        self.crossover_coeff = crossover_coeff  # low --> diverse parents
        self.high_is_good = high_is_good
        self._population = []
        self._population_history = []
        self._evaluation_history = []


    def _create_intial_population(self, population_size):
        return [self.create_member_fun() for _ in range(population_size)]


    def _step_population(self, population_size) -> list:

        # rank according to evaluation
        self._population.sort(key=lambda member: self.evaluation_fun(member), reverse=self.high_is_good)

        # select best members for elitism
        new_population = [self._population[i].copy() for i in range(int(self.p_elitism * population_size))]

        # apply crossover
        for _ in range(int(self.p_crossover * population_size)):
            member_indices = np.zeros(2)
            while (np.all(member_indices == member_indices[0]) or (max(member_indices) >= len(self._population))):
                member_indices = np.random.geometric(p=self.crossover_coeff, size=2)
            new_population.append(self.crossover_fun(self._population[member_indices[0]], self._population[member_indices[1]]))

        # apply mutations
        for member in new_population:
            if np.random.random() < self.p_mutate:
                new_population.append(self.mutate_fun(member))

        # fill up with random members
        for _ in range(population_size - len(new_population)):
            new_population.append(self.create_member_fun())

        self._population = new_population


    def run_evolution(self, n_generations, population_size, random_seed=None):
        # np.random.seed(random_seed)  #TODO: not working

        self._population = self._create_intial_population(population_size)
        self._population_history.append(self._population)
        self._evaluation_history.append(self.evaluation_fun(self.get_best_member()))

        print('generation ' + str(0).zfill(4) + ', best evaluation value: ' + str(self.evaluation_fun(self.get_best_member())))

        for generation_no in range(n_generations):
            self._step_population(population_size)
            self._population_history.append(self._population)
            self._evaluation_history.append(self.evaluation_fun(self.get_best_member()))
            print('generation ' + str(generation_no + 1).zfill(4) + ', best evaluation value: ' + str(self.evaluation_fun(self.get_best_member())))

    #TODO: to apply stepping of probabilities, create new run method and step values in between

    def get_current_population(self):
        return sorted(self._population, key=lambda member: self.evaluation_fun(member), reverse=self.high_is_good)

    def get_best_member(self):
        return sorted(self._population, key=lambda member: self.evaluation_fun(member), reverse=self.high_is_good)[0]

    def get_history(self):
        return (self._population_history, self._evaluation_history)

    def plot_evaluation_history(self):
        plt.plot(self._evaluation_history)
        plt.xlabel('generation'), plt.ylabel('best evaluation value')
        plt.xlim([0, len(self._evaluation_history)])
        plt.ylim([0, max(self._evaluation_history) * 1.1])
        plt.grid()
        plt.show()