import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from typing import NamedTuple, List

class Problem(NamedTuple):
    adjacency_matrix: np.ndarray
    graph: nx.Graph
    paths: List[List[int]]
    constraints: np.ndarray

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

    # reachability hack
    exists_valid_path = np.linalg.matrix_power(adjacency_matrix, num_vertices) > 0
    possible_source_targets = np.array(exists_valid_path.nonzero()).T
    source_targets_idx = np.random.choice(len(possible_source_targets), num_paths)
    source_targets = possible_source_targets[source_targets_idx]

    graph = nx.Graph(adjacency_matrix)

    paths = []
    for s,t in source_targets:
        path_generator = nx.all_simple_paths(graph, s, t)
        valid_paths = [
            x for x in path_generator
            if len(x) >= min_path_length and len(x) <= max_path_length
        ]
        assert valid_paths
        paths.append(valid_paths[np.random.choice(len(valid_paths))])

    constraints = np.random.uniform(min_constraint, max_constraint, adjacency_matrix.shape)

    return Problem(adjacency_matrix, graph, paths, constraints)


problem = gen_problem(7, 13, 3, 1, 5, 3, 3)
import pdb; pdb.set_trace()


