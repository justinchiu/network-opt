from pathlib import Path

from itertools import product

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

from typing import NamedTuple, List

class Problem(NamedTuple):
    graph: nx.Graph
    paths: List[List[int]]
    demand: np.ndarray
    constraints: np.ndarray
    r2p: np.ndarray
    r2p_tup: np.ndarray
    e2p: np.ndarray


basedir = Path("teavar/code/data/B4")
demand_file = basedir / "demand.txt"
nodes_file = basedir / "nodes.txt"
topology_file = basedir / "topology.txt"
path_file = basedir / "paths/FFC"

def process_demands(demand_file):
    with demand_file.open() as f:
        demands = [[float(d) for d in line.split()] for line in f]
    demands = np.array(demands).reshape((-1, 12, 12))
    return demands

def process_topology(topology_file):
    with topology_file.open() as f:
        f.readline()
        edges = []
        constraints = []
        for line in f:
            src, tgt, cap, p_failure = line.split()
            src = int(src) - 1
            tgt = int(tgt) - 1
            cap = float(cap)
            p_failure = float(p_failure)
            edges.append((src, tgt))
            constraints.append(cap)
        edges = np.array(edges)
        constraints = np.array(constraints)

        V = edges.max() + 1
        adjacency_constraints = np.zeros((V,V))
        adjacency_constraints[edges[:,0], edges[:,1]] = constraints

        return adjacency_constraints

def process_paths(path_file, demand):
    with path_file.open() as f:
        _, V = demand.shape

        r2p = np.zeros((V, V), dtype=np.int)
        P = 0
        for line in f:
            tokens = line.split()
            if not tokens:
                # empty line between path lists
                pass
            elif tokens[-1] == ":":
                # start of edge
                r_src = int(tokens[0][1:]) - 1
                r_tgt = int(tokens[2][1:]) - 1
            elif tokens[-2] == "@":
                # path internal
                assert(len(tokens) >= 5)
                r2p[r_src, r_tgt] += 1
            else:
                raise ValueError

        P = r2p.sum()
        # sparse version
        # r2p_tuple[s,t] = [start, end)
        r2p_tuple = np.zeros((V*V, 2), dtype=int)
        r2p_tuple[1:,0] = r2p.cumsum()[:-1]
        r2p_tuple[1:-1,1] = r2p_tuple[2:,0]
        r2p_tuple[-1,1] = P

        # dense version, assuming each request has the same number of paths
        Pk = r2p.max()

        # reset file
        f.seek(0)

        # binary tensor: whether a path passes through an edge
        e2p = np.zeros((V, V, P))
        p = 0
        for line in f:
            tokens = line.split()
            if not tokens:
                # empty line between path lists
                pass
            elif tokens[-1] == ":":
                # start of edge
                r_src = int(tokens[0][1:]) - 1
                r_tgt = int(tokens[2][1:]) - 1
            elif tokens[-2] == "@":
                # path internal
                assert(len(tokens) >= 5)

                for edge in tokens[1:-3]:
                    # middle edges
                    src, tgt, _  = edge.split(",")
                    e_src = int(src[2:]) - 1
                    e_tgt = int(tgt[1:-1]) - 1

                    e2p[e_src, e_tgt, p] = 1
                p += 1
            else:
                raise ValueError
        return r2p, r2p_tuple, e2p

def load_problem():
    demands = process_demands(demand_file)
    constraints = process_topology(topology_file)
    r2p, e2p = process_paths(path_file, demands[0])
    graph = nx.Graph(adjacency)

    return Problem(
        graph,
        paths,
        demands[0],
        constraints,
        r2p,
        e2p,
    )


problem = load_problem()

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

pi_r = np.zeros(())

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
