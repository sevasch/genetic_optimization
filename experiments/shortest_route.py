import os
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from genetic_optimizer import GeneticOptimizer
from functools import partial

''' CONSTANTS '''
COLORS = ['red', 'yellow', 'green', 'purple']

''' PARAMETERS '''
parser = argparse.ArgumentParser(description='Genetic optimization of route in graph. ')

# graph specific
parser.add_argument('--n_nodes', type=int, default=100, help='number of nodes in graph')
parser.add_argument('--x_lim', type=int, default=25, help='number of grid cells in x-direction')
parser.add_argument('--y_lim', type=int, default=25, help='number of grid cells in y-direction')
parser.add_argument('--lmbda', type=float, default=0.45, help='parameter for connections (low --> many connections)')

# optimizer specific
parser.add_argument('--n_generations', type=int, default=30, help='number of generations')
parser.add_argument('--population_size', type=int, default=30, help='number of members in population')
parser.add_argument('--p_elitism', type=float, default=0.1, help='proportion of elitist members')
parser.add_argument('--p_crossover', type=float, default=0.3, help='proportion of crossover members')
parser.add_argument('--p_mutate', type=float, default=0.1, help='probability to create mutated copy of member')

# output
parser.add_argument('--save_frames', type=bool, default=True, help='wether to save frames or not')
parser.add_argument('--save_dir', type=str, default='frames', help='where to save frames')

args = parser.parse_args()


''' FUNCTIONS '''
def make_edge_list(graph_matrix):
    indices = np.where(graph_matrix > 0)
    return [[indices[0][i], indices[1][i]] for i in range(len(indices[0]))]

class Environment():
    def __init__(self, n_nodes, x_lim, y_lim, lmbda):
        self.n_nodes = n_nodes
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.lmbda = lmbda
        self.graph_matrix, self.node_positions, self.edge_list = self._generate_graph()

    def _generate_graph(self):
        # generate node positions
        node_positions = np.zeros((self.n_nodes, 2))
        for i in range(self.n_nodes):
            already_exists = True
            counter = 0
            while already_exists:
                node_position = np.array((np.random.randint(0, self.x_lim), np.random.randint(0, self.y_lim)))
                if not (node_position == node_positions).all(1).any():
                    already_exists = False
                counter += 1
                if counter > 1000:
                    continue
            node_positions[i] = node_position

        # generate node connections
        graph_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for node_no, node_pos in enumerate(node_positions):
            distances = np.linalg.norm(node_positions - node_pos, axis=1)
            distances[distances == 0] = np.inf
            nearest_connection_index = np.argmin(distances)
            graph_matrix[node_no, nearest_connection_index] = np.min(distances[distances > 0])
            probability_for_connection = np.exp(-self.lmbda * distances)
            additional_connection_indices = np.where(probability_for_connection > np.random.uniform(0, 1, self.n_nodes))[0]
            for connection_index in additional_connection_indices:
                if not connection_index == node_no:
                    graph_matrix[node_no, connection_index] = distances[connection_index]
                    graph_matrix[connection_index, node_no] = distances[connection_index]

        edge_list = make_edge_list(graph_matrix)

        return graph_matrix, node_positions, edge_list

    def draw(self, color='cyan'):
        map_graph = nx.Graph()
        map_graph.add_edges_from(self.edge_list)
        nx.draw(map_graph, self.node_positions, node_color=color, with_labels=True)


def generate_route_segment(graph_matrix, node_start: int, node_end: int):
    route_segment = np.ones(1, dtype=int) * node_start
    i = 0
    while not route_segment[i] == node_end:
        route_segment = np.append(route_segment, np.random.choice(np.nonzero(graph_matrix[route_segment[i]])[0]))
        i += 1

    return route_segment

def get_distance(env, route):
    distance = 0
    for i in range(len(route) - 1):
        distance += np.linalg.norm(env.node_positions[route[i]] - env.node_positions[route[i + 1]])
    return distance


def mutate(env, route):
    index_start = np.random.randint(0, len(route) - 1)
    index_end = index_start + np.random.randint(0, len(route) - index_start)
    route_before = route[:index_start]
    route_after = route[index_end + 1:]
    route_intermediate = generate_route_segment(env.graph_matrix, route[index_start], route[index_end])
    mutated_route = np.concatenate([route_before, route_intermediate, route_after])
    return mutated_route

def crossover(first_route, second_route):
    intersections = np.intersect1d(first_route[1:-1], second_route[1:-1])
    if len(intersections):
        intersection = np.random.choice(intersections)
        own_snipping_location = np.random.choice(np.where(first_route[1:-1] == intersection)[0]) + 1
        other_snipping_location = np.random.choice(np.where(second_route[1:-1] == intersection)[0]) + 1
        if np.random.random() > 0.5:
            new_route = np.concatenate([first_route[:own_snipping_location], second_route[other_snipping_location:]])
        else:
            new_route = np.concatenate([second_route[:other_snipping_location], first_route[own_snipping_location:]])
    else:
        if np.random.random() > 0.5:
            new_route = first_route
        else:
            new_route = second_route
    return new_route

def draw_route(env, route, color='red'):
    edges = [[route[i], route[i + 1]] for i in range(len(route) - 1)]
    route_graph = nx.Graph()
    route_graph.add_edges_from(edges)
    nx.draw(route_graph, env.node_positions, node_color=color, with_labels=True)


def main():
    np.random.seed(5)

    # create environment
    env = Environment(args.n_nodes, args.x_lim, args.y_lim, args.lmbda)

    # create optimizer
    optimizer = GeneticOptimizer(create_member_fun = partial(generate_route_segment, env.graph_matrix, 0, np.shape(env.graph_matrix)[0] - 1),
                                 mutate_fun = partial(mutate, env),
                                 crossover_fun = crossover,
                                 evaluation_fun = partial(get_distance, env),
                                 p_elitism = args.p_elitism,
                                 p_crossover = args.p_crossover,
                                 p_mutate = args.p_mutate)

    # run evolution, plot results and extract population
    optimizer.run_evolution(args.n_generations,
                            args.population_size)

    optimizer.plot_evaluation_history()

    population_history, fitness_history = optimizer.get_history()


    # visualize reesults
    if args.save_frames:
        os.makedirs(args.save_dir, exist_ok=True)

    plt.figure(figsize=[12, 8])
    env.draw()
    for generation_no, population in enumerate(population_history):
        plt.clf()
        for route in population[:1]:
            env.draw()
            draw_route(env, route, color=COLORS[np.mod(generation_no, len(COLORS))])
            plt.text(args.x_lim * 0.95, args.y_lim * 0.01, 'gen {}'.format(generation_no))
            plt.pause(0.2)
        if args.save_frames:
            plt.savefig(os.path.join(args.save_dir, 'gen' + str(generation_no).zfill(int(np.log10(args.n_generations)) + 1) + '.png'))
    plt.show()


if '__main__' == __name__:
    main()