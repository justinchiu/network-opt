import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

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
    constraints = constraints * adjacency_matrix
    # by defn, have constraint of 0 if there is no edge

    return Problem(adjacency_matrix, graph, paths, constraints)


problem = gen_problem(7, 13, 3, 1, 5, 3, 3)

G = problem.graph
pos = nx.spring_layout(G, seed=225)  # Seed for reproducible layout
nx.draw(G, pos, with_labels=True)
#node_labels = nx.get_node_attributes(G,'state')
#nx.draw_networkx_labels(G, pos, labels = node_labels)
#edge_labels = nx.get_edge_attributes(G,'state')
#nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

plt.savefig('example_graph.png')
#plt.show()

K = len(problem.paths)
V = problem.constraints.shape[0]
pi_e = np.zeros((V, V, K))

for p, path in enumerate(problem.paths):
    for s,t in zip(path, path[1:]):
        pi_e[s,t,p] = 1

# CVXPY
X = cp.Variable(K)
objective = cp.Minimize(-cp.sum(X))
constraints = [
    pi_e.reshape(V*V, K) @ X <= problem.constraints.reshape(V*V),
    X >= 0,
]

prob = cp.Problem(objective, constraints)
result = prob.solve()
print(X.value)
print(constraints[0].dual_value)



# ADMM

def naive_admm_solver(
    problem,
    pi_e,
    rho,
    x, z, s,
    lambda1, lambda2,
    num_iters = 100,
):
    c = problem.constraints.reshape(-1)

    path_lens = np.array([len(p) for p in problem.paths])

    L = sum(path_lens)

    pi_e_flat = pi_e.reshape(V*V, K).astype(int)
    num_paths_for_edge = pi_e_flat.sum(-1)

    pi_e_flat_bool = pi_e.reshape(-1).astype(bool)

    A_invs = [
        np.linalg.inv(np.ones((k, k)) + np.eye(k))
        if k > 0 else None
        for k in num_paths_for_edge.tolist()
    ]
    As = [
        np.ones((k, k)) + np.eye(k)
        if k > 0 else None
        for k in num_paths_for_edge.tolist()
    ]

    for iter in range(num_iters):
        # x update
        #x_numerator = 1 - (lambda2 * pi_e_flat).sum(0) + rho * (pi_e_flat * z).sum(0)
        x_numerator = (
            1 + lambda1 - (lambda4 * pi_e_flat).sum(0)
            + rho * ()
            + rho * (pi_e_flat * z).sum(0)
        )
        #x_denominator = (path_lens-1) * rho
        x_denominator = path_lens * rho
        x = np.maximum(0, x_numerator / x_denominator)

        # z update
        for e in range(c.size):
            if A_invs[e] is not None:
                A_inv = A_invs[e] / rho
                path_ix = pi_e_flat[e].astype(bool)
                #b = (-lambda1[e,None] - lambda2[e, path_ix] + lambda5[e, path_ix]
                b = (-lambda1[e,None] - lambda2[e, path_ix]
                    + rho * (-c[e,None] + s[e,None] - x[path_ix]))
                z[e, path_ix] = -A_inv @ b
        #z = np.maximum(0, z)
        #import pdb; pdb.set_trace()
         
        # s update
        s = np.maximum(0,
            #(lambda1 - lambda4 + rho * (c - (pi_e_flat * z).sum(-1))) / (2 * rho))
            (lambda1 + rho * (c - (pi_e_flat * z).sum(-1))) / (rho))

        r1 = c - (pi_e_flat * z).sum(-1) - s
        r2 = x[None] - z

        # lambda update
        lambda1 += rho * r1

        lambda2 = lambda2.reshape(-1)
        lambda2[pi_e_flat_bool] += rho * r2.reshape(-1)[pi_e_flat_bool]
        #np.add.at(lambda2, pi_e_flat, rho * r2[pi_e_flat])
        lambda2 = lambda2.reshape(pi_e_flat.shape)

    return x, z, s, lambda1, lambda2


s = np.zeros((V,V))
#np.copyto(s, problem.constraints)
s = s.reshape(-1)

x, z, s, l1, l2 = naive_admm_solver(
    problem,
    pi_e,
    1,
    np.zeros((K,)), np.zeros((V*V, K)), s,
    np.zeros((V*V,)), np.zeros((V*V, K)),
    num_iters = 1000,
)
print(s)

import pdb; pdb.set_trace()
