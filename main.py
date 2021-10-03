import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

N_NODES = 100
X_LIM = 50
Y_LIM = 50
LMBDA = 0.3

POPULATION_SIZE = 50
N_GENERATIONS = 50
N_ELITISM = 15
N_RANDOM = 5
MUTATION_PROBABILITY = 0.02

COLORS = ['red', 'yellow', 'green', 'purple']

def create_intial_population(n_members):
    return [Member() for _ in range(n_members)]

def generate_new_population(p_elitism, p_direct_mutation, p_crossover, p_crossover_mutate) -> list:
    assert p_elitism + p_direct_mutation + p_crossover + p_crossover_mutate <= 1
    new_population = []
    return new_population


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

class Member():
    def __init__(self, graph_matrix, node_positions, route=None):
        self.graph_matrix = graph_matrix
        self.node_positions = node_positions
        if route is not None:
            self.route = route
        else:
            self.route = self._generate_route_segment(0, np.shape(graph_matrix)[0] - 1)

    def _generate_route_segment(self, node_start: int, node_end: int):
        route_segment = np.ones(1, dtype=int) * node_start
        i = 0
        while not route_segment[i] == node_end:
            route_segment = np.append(route_segment, np.random.choice(np.nonzero(self.graph_matrix[route_segment[i]])[0]))
            i += 1

        return route_segment

    def get_total_distance(self):
        distance = 0
        for i in range(len(self.route) - 1):
            distance += np.linalg.norm(self.node_positions[self.route[i]] - self.node_positions[self.route[i + 1]])
        return distance

    def mutate(self):
        index_start = np.random.randint(0, len(self.route) - 1)
        index_end = index_start + np.random.randint(0, len(self.route) - index_start)
        route_before = self.route[:index_start]
        route_after = self.route[index_end + 1:]
        route_intermediate = self._generate_route_segment(self.route[index_start], self.route[index_end])
        mutated_route = np.concatenate([route_before, route_intermediate, route_after])
        return Member(self.graph_matrix, self.node_positions, mutated_route)

    def crossover_with(self, other):
        own_route = self.route
        other_route = other.route
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
        return Member(self.graph_matrix, self.node_positions, new_route)


    def draw(self, color='red'):
        edges = [[self.route[i], self.route[i + 1]] for i in range(len(self.route) - 1)]
        route_graph = nx.Graph()
        route_graph.add_edges_from(edges)
        nx.draw(route_graph, self.node_positions, node_color=color, with_labels=True)


if '__main__' == __name__:
    np.random.seed(6)

    # create environment
    env = Environment(N_NODES, X_LIM, Y_LIM, LMBDA)

    # create initial population
    members = []
    for _ in range(POPULATION_SIZE):
        members.append(Member(env.graph_matrix, node_positions=env.node_positions))
    pass
    print('created generation 0')

    best_distance = []
    for generation_no in range(N_GENERATIONS):
        # rate according to distance
        members.sort(key=lambda member: member.get_total_distance(), reverse=False)

        # select best members for elitism
        new_members = [members[i] for i in range(N_ELITISM)]

        # create random members
        for _ in range(N_RANDOM):
            new_members.append(Member(env.graph_matrix, env.node_positions))

        # do crossover
        for _ in range(POPULATION_SIZE - N_ELITISM - N_RANDOM):
            new_members.append(np.random.choice(members).crossover_with(np.random.choice(members)))

        # apply mutations
        for i in range(len(new_members)):
            if np.random.random() < MUTATION_PROBABILITY:
                new_members[i] = new_members[i].mutate()

        # update generation
        members = new_members

        print('created generation {}'.format(generation_no + 1))
        best_distance.append(min([m.get_total_distance() for m in new_members]))

        # plot top 5 members
        plt.clf()
        for member in sorted(members, key=lambda m: m.get_total_distance())[:5]:
            env.draw()
            member.draw(color=COLORS[np.mod(generation_no, len(COLORS))])
            plt.pause(0.2)


    # draw best
    # env.draw()
    # sorted(members, key=lambda m: m.get_total_distance())[0].draw()

    plt.figure()
    plt.plot(best_distance)
    plt.xlabel('generation'), plt.ylabel('best distance')
    plt.show()