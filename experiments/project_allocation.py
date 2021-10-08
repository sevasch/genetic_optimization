import numpy as np
import csv
from genetic_optimizer import GeneticOptimizer
from functools import partial

N_PRIORITIES = 3
N_STUDENTS = 9
NO_MATCH_PENALTY = 20

N_GENERATIONS = 300
POPULATION_SIZE = 100
P_ELITISM = 0.1
P_CROSSOVER = 0.7
P_MUTATE = 0.2


def create_random(priority_list, n_projects):
    assert n_projects > len(priority_list[0])
    project_list = np.arange(n_projects)
    return list(np.random.choice(project_list, len(priority_list), replace=False))


def get_ranking(priority_list: list, chromosome: list):
    total_score = 0
    for student_project, student_priorities in zip(chromosome, priority_list):
        position_on_list = np.where(np.array(student_priorities) == student_project)[0]
        total_score += position_on_list.item() if len(position_on_list) > 0 else NO_MATCH_PENALTY
    return total_score


def mutate(n_projects, chromosome):
    new_chromosome = chromosome.copy()
    if np.random.random() < 0.5:
        indices = np.random.choice(np.arange(len(chromosome)), 2, replace=False)
        new_chromosome[indices[0]] = chromosome[indices[1]].copy()
        new_chromosome[indices[1]] = chromosome[indices[0]].copy()
    else:
        index = np.random.choice(np.arange(len(chromosome)))
        vacant_projects = set(np.arange(n_projects)).difference(set(chromosome))
        new_chromosome[index] = np.random.choice(list(vacant_projects)).copy()
    return new_chromosome


def crossover(first_chromosome, second_chromosome):
    new_chromosome = first_chromosome.copy()
    intersection = np.intersect1d(first_chromosome, second_chromosome)
    if intersection.any():
        first_value = np.random.choice(intersection)
        # print(own_value, first_chromosome)
        if first_value in second_chromosome:
            second_value = second_chromosome[np.where(second_chromosome == first_value)[0].item()]
            if second_value in first_chromosome:
                first_index = np.where(first_chromosome == first_value)[0].item()
                second_index = np.where(first_chromosome == second_value)[0].item()
                new_chromosome[first_index] = second_value.copy()
                new_chromosome[second_index] = first_value.copy()
    return new_chromosome


if '__main__' == __name__:
    np.random.seed(1)

    # generate random priority list
    # priority_list = [np.random.choice(np.arange(N_PROJECTS), N_PRIORITIES, replace=False).tolist() for _ in range(N_STUDENTS)]

    # input priority list from file
    names = []
    priority_list = []
    with open('priority_list.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for line_no, row in enumerate(csv_reader):
            if line_no > 0:
                names.append(row[0])
                priority_list.append([int(project_no) for project_no in row[1:4]])
    N_PROJECTS = 30
    N_STUDENTS = len(priority_list)
    print(N_PROJECTS, N_STUDENTS)

    # create optimizer
    optimizer = GeneticOptimizer(create_member_fun=partial(create_random, priority_list, N_PROJECTS),
                                 mutate_fun=partial(mutate, N_PROJECTS),
                                 crossover_fun=crossover,
                                 evaluation_fun=partial(get_ranking, priority_list),
                                 p_elitism=P_ELITISM,
                                 p_crossover=P_CROSSOVER,
                                 p_mutate=P_MUTATE,
                                 high_is_good=False)

    optimizer.run_evolution(N_GENERATIONS, POPULATION_SIZE)
    optimizer.plot_evaluation_history()
    population_history, fitness_history = optimizer.get_history()

    # for result, priorities in zip(optimizer.get_best_member(), priority_list):
    #     print(priorities, result)

    for name, result, priorities in zip(names, optimizer.get_best_member(), priority_list):
        print('Student: {}, \t priorities: {}, \t gets project {}'.format(name,  priorities, result))