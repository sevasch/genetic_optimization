import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from member import Member
from genetic_optimizer import GeneticOptimizer

N_NODES = 100
X_LIM = 50
Y_LIM = 50
LMBDA = 0.30

N_GENERATIONS = 100
POPULATION_SIZE = 10
P_ELITISM = 0.2
P_CROSSOVER = 0.2
P_MUTATE = 0.2

COLORS = ['red', 'yellow', 'green', 'purple']


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
            while already_exists:
                node_position = np.array((np.random.randint(0, self.x_lim), np.random.randint(0, self.y_lim)))
                if not (node_position == node_positions).all(1).any():
                    already_exists = False
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

class Route(Member):
    def __init__(self, route, graph_matrix, node_positions):
        super().__init__(route)
        self.graph_matrix = graph_matrix
        self.node_positions = node_positions

    @classmethod
    def create_random(cls, graph_matrix, node_positions):
        return cls(cls._generate_route_segment(graph_matrix, 0, np.shape(graph_matrix)[0] - 1), graph_matrix, node_positions)

    @staticmethod
    def _generate_route_segment(graph_matrix, node_start: int, node_end: int):
        route_segment = np.ones(1, dtype=int) * node_start
        i = 0
        while not route_segment[i] == node_end:
            route_segment = np.append(route_segment, np.random.choice(np.nonzero(graph_matrix[route_segment[i]])[0]))
            i += 1

        return route_segment

    def _get_total_distance(self):
        distance = 0
        for i in range(len(self.chromosome) - 1):
            distance += np.linalg.norm(self.node_positions[self.chromosome[i]] - self.node_positions[self.chromosome[i + 1]])
        return distance

    def get_fitness(self) -> float:
        return self._get_total_distance()

    def mutate(self):
        index_start = np.random.randint(0, len(self.chromosome) - 1)
        index_end = index_start + np.random.randint(0, len(self.chromosome) - index_start)
        route_before = self.chromosome[:index_start]
        route_after = self.chromosome[index_end + 1:]
        route_intermediate = self._generate_route_segment(self.graph_matrix, self.chromosome[index_start], self.chromosome[index_end])
        mutated_route = np.concatenate([route_before, route_intermediate, route_after])
        return Route(mutated_route, self.graph_matrix, self.node_positions)


    def crossover(self, other):
        own_route = self.chromosome
        other_route = other.chromosome
        intersections = np.intersect1d(own_route[1:-1], other_route[1:-1])
        if len(intersections):
            intersection = np.random.choice(intersections)
            own_snipping_location = np.random.choice(np.where(own_route[1:-1] == intersection)[0]) + 1
            other_snipping_location = np.random.choice(np.where(other_route[1:-1] == intersection)[0]) + 1
            if np.random.random() > 0.5:
                new_route = np.concatenate([own_route[:own_snipping_location], other_route[other_snipping_location:]])
            else:
                new_route = np.concatenate([other_route[:other_snipping_location], own_route[own_snipping_location:]])
        else:
            if np.random.random() > 0.5:
                new_route = own_route
            else:
                new_route = other_route
        return Route(new_route, self.graph_matrix, self.node_positions)

    def draw(self, color='red'):
        edges = [[self.chromosome[i], self.chromosome[i + 1]] for i in range(len(self.chromosome) - 1)]
        route_graph = nx.Graph()
        route_graph.add_edges_from(edges)
        nx.draw(route_graph, self.node_positions, node_color=color, with_labels=True)


if '__main__' == __name__:
    np.random.seed(6)

    # create environment
    env = Environment(N_NODES, X_LIM, Y_LIM, LMBDA)

    route = Route.create_random(env.graph_matrix, env.node_positions)

    # create optimizer
    optimizer = GeneticOptimizer(Route,
                                 P_ELITISM,
                                 P_CROSSOVER,
                                 P_MUTATE,
                                 env.graph_matrix,
                                 env.node_positions)

    optimizer.run_evolution(N_GENERATIONS,
                            POPULATION_SIZE)

    optimizer.plot_fitness_history()

    population_history, fitness_history = optimizer.get_history()


    env.draw()

    for generation_no, population in enumerate(population_history):
        plt.clf()
        for member in population[:1]:
            env.draw()
            member.draw(color=COLORS[np.mod(generation_no, len(COLORS))])
            plt.text(X_LIM*0.95, Y_LIM * 0.01, 'gen {}'.format(generation_no))
            plt.pause(0.2)
        # plt.savefig('plots/gen' + str(generation_no).zfill(3) + '.png')

    plt.show()