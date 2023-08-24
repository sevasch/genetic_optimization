import numpy as np
import csv
from src.genetic_optimizer import genetic_optimizer
import matplotlib.pyplot as plt
from functools import partial

#

N_PROJECTS = 30
NO_MATCH_PENALTY = 20

N_GENERATIONS = 300
POPULATION_SIZE = 100
P_ELITISM = 0.2
P_CROSSOVER = 0.3
P_MUTATE = 0.3


def create_random(priority_list, n_projects):
    assert n_projects > len(priority_list[0])
    project_list = np.arange(n_projects)
    return list(np.random.choice(project_list, len(priority_list), replace=False))


def get_ranking(priority_list: list, chromosome: list):
    total_score = 0
    for student_project, student_priorities in zip(chromosome, priority_list):
        position_on_list = np.where(np.array(student_priorities) == student_project)[0]
        total_score += (
            position_on_list.item() if len(position_on_list) > 0 else NO_MATCH_PENALTY
        )
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
            second_value = second_chromosome[
                np.where(second_chromosome == first_value)[0].item()
            ]
            if second_value in first_chromosome:
                first_index = np.where(first_chromosome == first_value)[0].item()
                second_index = np.where(first_chromosome == second_value)[0].item()
                new_chromosome[first_index] = second_value.copy()
                new_chromosome[second_index] = first_value.copy()
    return new_chromosome


if "__main__" == __name__:
    # np.random.seed(1)

    # generate random priority list
    # priority_list = [np.random.choice(np.arange(N_PROJECTS), N_PRIORITIES, replace=False).tolist() for _ in range(N_STUDENTS)]

    # input priority list from file
    names = []
    priority_list = []
    with open("example_data/priority_list.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=";")
        for line_no, row in enumerate(csv_reader):
            if line_no == 0:
                N_PRIORITIES = len(row) - 1
            else:
                names.append(row[0])
                priority_list.append([int(project_no) for project_no in row[1:]])
    N_STUDENTS = len(priority_list)
    print(
        "Anzahl Projekte: {}, Anzahl Studenten: {}, Anzahl wählbare Prioritäten: {}".format(
            N_PROJECTS, N_STUDENTS, N_PRIORITIES
        )
    )

    # Run optimzation
    best_member, population_history, fitness_history = genetic_optimizer(
        fn_create_random_member=partial(create_random, priority_list, N_PROJECTS),
        fn_mutate=partial(mutate, N_PROJECTS),
        fn_crossover=crossover,
        fn_cost=partial(get_ranking, priority_list),
        p_elitism=P_ELITISM,
        p_crossover=P_CROSSOVER,
        p_mutate=P_MUTATE,
        n_generations=N_GENERATIONS,
        population_size=POPULATION_SIZE,
        crossover_coeff=0.2
    )

    # Display history of fitness values
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(N_GENERATIONS + 1), fitness_history)
    ax.grid()
    ax.set_xlabel("Evolution Step")
    ax.set_ylabel("Cost")
    ax.set_xlim(0, N_GENERATIONS + 1)
    plt.show()


    matches = N_PRIORITIES * [0]
    for name, result, priorities in zip(
        names, best_member, priority_list
    ):
        position_in_priorities = np.where(result == priorities)[0]
        if len(position_in_priorities):
            priority = position_in_priorities.item()
            if priority < N_PRIORITIES:
                matches[priority] += 1
        print(
            "Student: {}, \t priorities: {}, \t gets project {}, \t which was priority {}".format(
                name, priorities, result, priority + 1
            )
        )

    labels = ["Prio {}".format(i + 1) for i in range(N_PRIORITIES)] + ["none"]
    plt.title("distribution of allocated priorities")
    plt.bar(labels, matches + [N_STUDENTS - np.sum(matches)])
    plt.show()
