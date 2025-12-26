import networkx as nx
import numpy as np


def generate_custom_sbm(n, h, k,  a, b, c, d, seed=None):
    np.random.seed(seed)
    group_sizes = [n // h] * h
    cluster_sizes = [n // k] * k

    clusters = []
    for i, size in enumerate(cluster_sizes):
        clusters.extend([i] * size)

    groups = []
    for i in range(k):
        for j in range(h):
            groups.extend([j] * ((n//k)//h))

    # np.random.shuffle(groups)
    # np.random.shuffle(clusters)
    group_membership = {i: groups[i] for i in range(n)}
    cluster_membership = {i: clusters[i] for i in range(n)}

    adj_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if clusters[i] == clusters[j] and groups[i] == groups[j]:
                p = a
            elif clusters[i] != clusters[j] and groups[i] == groups[j]:
                p = b
            elif clusters[i] == clusters[j] and groups[i] != groups[j]:
                p = c
            else:
                p = d

            if np.random.rand() < p:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    G = nx.from_numpy_array(adj_matrix)
    nx.set_node_attributes(G, group_membership, 'group')
    nx.set_node_attributes(G, cluster_membership, 'cluster')
    nx.set_node_attributes(G,group_membership, "protected")

    return G

