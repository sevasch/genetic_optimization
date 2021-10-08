import numpy as np
import matplotlib.pyplot as plt
from member import Member
from genetic_optimizer import GeneticOptimizer

N_PROJECTS = 5
N_PRIORITIES = 3
N_STUDENTS = 3

N_GENERATIONS = 100
POPULATION_SIZE = 10
P_ELITISM = 0.2
P_CROSSOVER = 0.
P_MUTATE = 0.


class Allocation(Member):
    def __init__(self, route, priority_list):
        super().__init__(route)
        self.priority_list = priority_list

    @classmethod
    def create_random(cls, priority_list, n_projects):
        assert n_projects > len(priority_list)
        project_list = np.arange(n_projects)
        return cls(np.random.choice(project_list, len(priority_list), replace=False), priority_list)

    def get_fitness(self) -> float:
        total_score = 0
        for student_project, student_priorities in zip(self.chromosome, self.priority_list):
            position_on_list = np.where(student_project == student_priorities)[0]
            total_score += position_on_list.item() if position_on_list.any() else 4
        return total_score

    def mutate(self):
        indices = np.random.choice(np.arange(len(self.chromosome)), 2, replace=False)
        new_chromosome = self.chromosome.copy()
        new_chromosome[indices[0]] = self.chromosome[indices[1]].copy()
        new_chromosome[indices[1]] = self.chromosome[indices[0]].copy()
        return self.__class__(new_chromosome, self.priority_list)

    def crossover(self, other):
        own_chromosome = self.chromosome
        other_chromosome = other.chromosome
        new_chromosome = own_chromosome.copy()
        own_value = np.random.choice(np.intersect1d(own_chromosome, other_chromosome))
        if own_value.any():
            index1 = np.where(own_chromosome == own_value)[0].item()
            other_value = other_chromosome[index1]
            if other_value in own_chromosome:  #TODO: find a better solution
                index2 = np.where(own_chromosome == other_value)[0].item()
                new_chromosome[index1] = other_chromosome[index1]
                new_chromosome[index2] = other_chromosome[index2]
        return self.__class__(new_chromosome, self.priority_list)



if '__main__' == __name__:
    np.random.seed(1)
    priority_list = [np.random.choice(np.arange(N_PROJECTS), N_PRIORITIES, replace=False).tolist() for _ in range(N_STUDENTS)]

    # create optimizer
    optimizer = GeneticOptimizer(Allocation,
                                 P_ELITISM,
                                 P_CROSSOVER,
                                 P_MUTATE,
                                 priority_list,
                                 N_PROJECTS)

    optimizer.run_evolution(N_GENERATIONS,
                            POPULATION_SIZE)

    optimizer.plot_fitness_history()

    population_history, fitness_history = optimizer.get_history()

    for result, priorities in zip(optimizer.get_best_member().chromosome, priority_list):
        print(priorities, result)
