import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("Genetic Optimization Logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def genetic_optimizer(
     fn_create_random_member: callable,
     fn_mutate: callable,
     fn_crossover: callable,
     fn_cost: callable,
     p_elitism: float,
     p_crossover: float,
     p_mutate: float,
     crossover_coeff: float,
     n_generations: int, 
     population_size: int
): 
    """Run genetic optimization algorithm. 

    Args:
        fn_create_random_member: Create randomized member. 
        fn_mutate:               Mutate mem
        fn_crossover: _description_
        fn_cost: _description_
        p_elitism: _description_
        p_crossover: _description_
        p_mutate: _description_
        crossover_coeff: _description_
        n_generations: _description_
        population_size: _description_

    Returns:
        _description_
    """
    assert p_elitism + p_crossover + p_mutate <= 1

    def get_best_member():
        return sorted(population, key=lambda member: fn_cost(member))[0]

    population = []
    population_history = []
    fitness_history = []
    
    # Initialize population
        
    population = [fn_create_random_member() for _ in range(population_size)]
    population_history.append(population)
    fitness_history.append(fn_cost(get_best_member()))

    logger.info("Optimization started!")
    logger.info("Generation " + str(0).rjust(5) + ", best evaluation value: " + str(fn_cost(get_best_member())))

    for generation_no in range(n_generations):
        # rank according to evaluation
        population.sort(key=lambda member: fn_cost(member))

        # select best members for elitism
        new_population = [population[i].copy() for i in range(int(p_elitism * population_size))]

        # apply crossover
        for _ in range(int(p_crossover * population_size)):
            member_indices = np.zeros(2)
            while (np.all(member_indices == member_indices[0]) or (max(member_indices) >= len(population))):
                member_indices = np.random.geometric(p=crossover_coeff, size=2)
            new_population.append(fn_crossover(population[member_indices[0]], population[member_indices[1]]))

        # apply mutations
        for member in new_population:
            if np.random.random() < p_mutate:
                new_population.append(fn_mutate(member))

        # fill up with random members
        for _ in range(population_size - len(new_population)):
            new_population.append(fn_create_random_member())

        population_history.append(population)
        fitness_history.append(fn_cost(get_best_member()))
        logger.info("Generation " + str(generation_no + 1).rjust(5) + ", best evaluation value: " + str(fn_cost(get_best_member())))

        population = new_population

    print("Evolution completed after {} generations!".format(generation_no + 1))

    return get_best_member(), population_history, fitness_history
