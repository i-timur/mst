import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def init_graph(num_nodes):
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            adjacency_matrix[i][j] = adjacency_matrix[j][i] = np.random.randint(0, 10)
    return adjacency_matrix


def prim(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    dist = np.ones(n) * np.inf
    parents = [-1] * n
    is_visited = [False] * n

    start = 0
    dist[start] = 0

    for _ in range(n):
        u = -1
        for i in range(n):
            if not is_visited[i] and (u == -1 or dist[i] < dist[u]):
                u = i
        is_visited[u] = True

        for v in range(n):
            if not is_visited[v] and adjacency_matrix[u, v] > 0 and adjacency_matrix[u, v] < dist[v]:
                dist[v] = adjacency_matrix[u, v]
                parents[v] = u

    min_spanning_tree = np.zeros((n, n))
    for v in range(1, n):
        u = parents[v]
        min_spanning_tree[u, v] = min_spanning_tree[v, u] = adjacency_matrix[u, v]

    return nx.from_numpy_array(min_spanning_tree)


def remove_max(graph, num_edges_to_remove):
    adjacency_matrix = nx.to_numpy_array(graph)

    edges = []
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            weight = adjacency_matrix[i, j]
            if weight > 0:
                edges.append((i, j, weight))

    edges.sort(key=lambda x: -x[2])

    for i in range(min(num_edges_to_remove, len(edges))):
        edge = edges[i]
        adjacency_matrix[edge[0], edge[1]] = 0
        adjacency_matrix[edge[1], edge[0]] = 0

    return nx.from_numpy_array(adjacency_matrix)


def draw(graph):
    pos = nx.spring_layout(graph)
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw(graph, pos, node_size=100)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=10, font_family="sans-serif", font_color="b")
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family="sans-serif")
    plt.show()


if __name__ == '__main__':
    n = 4

    adjacency_matrix = init_graph(n)
    G = nx.from_numpy_array(adjacency_matrix)
    draw(G)

    G = prim(nx.to_numpy_array(G))
    draw(G)

    G = remove_max(G, 3)
    draw(G)
