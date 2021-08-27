import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def gen_problem(
    num_vertices, num_edges, num_paths,
    min_path_length=5, max_path_length=5,
    min_constraint=3, max_constraint=3,
):
    edges = np.random.permutation(int(num_vertices * (num_vertices - 1) / 2))[:num_edges]

    adjacency_matrix = np.zeros((num_vertices, num_vertices))
    ix, iy = np.triu_indices(num_vertices)
    adjacency_matrix[ix[edges], iy[edges]] = 1
    adjacency_matrix.T[ix[edges], iy[edges]] = 1
    np.fill_diagonal(adjacency_matrix, 0)

    exists_valid_path = np.linalg.matrix_power(adjacency_matrix, num_vertices) > 0
    possible_source_targets = np.array(exists_valid_path.nonzero()).T
    source_targets_idx = np.random.choice(len(possible_source_targets), num_paths)
    source_targets = possible_source_targets[source_targets_idx]

    paths = []
    for s,t in source_targets:
        path_length = np.random.choice(np.arange(min_path_length, max_path_length+1))


gen_problem(7, 13, 3, 3, 3, 3, 3)
