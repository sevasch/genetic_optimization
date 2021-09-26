import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

N_NODES = 10
X_LIM = 10
Y_LIM = 10
LMBDA = 0.3

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

        edge_list = make_edge_list(graph_matrix)

        return graph_matrix, node_positions, edge_list

    def draw(self):
        map_graph = nx.Graph()
        map_graph.add_edges_from(self.edge_list)
        nx.draw(map_graph, self.node_positions, node_color='blue', )
        plt.show()

if '__main__' == __name__:

    env = Environment(N_NODES, X_LIM, Y_LIM, LMBDA)
    env.draw()

