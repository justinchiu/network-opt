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
    demand: np.ndarray # in adjacency matrix form
    constraints: np.ndarray # in adjacency matrix form
    #r2p: np.ndarray
    #r2p_tup: np.ndarray
    P: int
    Pr: int
    e2p: np.ndarray # binary map from edge to which paths appear
    p2e: np.ndarray # binary map from path to included edges


basedir = Path("teavar/code/data/B4")
demand_file = basedir / "demand.txt"
nodes_file = basedir / "nodes.txt"
topology_file = basedir / "topology.txt"
path_file = basedir / "paths/FFC"

scale = 1000
#scale = 1

def process_demands(demand_file):
    with demand_file.open() as f:
        demands = [[float(d) for d in line.split()] for line in f]
    demands = np.array(demands).reshape((-1, 12, 12))
    return demands[0] / scale

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

        return adjacency_constraints / scale

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

        # sparse version
        # r2p_tuple[s,t] = [start, end)
        r2p_tuple = np.zeros((V*V, 2), dtype=int)
        r2p_tuple[1:,0] = r2p.cumsum()[:-1]
        r2p_tuple[1:-1,1] = r2p_tuple[2:,0]
        r2p_tuple[-1,1] = P

        Pr = r2p.max()
        #P = r2p.sum()
        P = V * V * Pr

        # reset file
        f.seek(0)

        requests = []

        # binary tensor: whether a path passes through an edge
        e2p = np.zeros((V, V, P))
        p2e = np.zeros((P, V, V))
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

                if r_src == r_tgt - 1:
                    # add Pr null paths for diagonal entries
                    # WARNING: assumes that every src-tgt pair / request has paths
                    p += Pr
                requests.append([])
            elif tokens[-2] == "@":
                # path internal
                assert(len(tokens) >= 5)

                edges = []
                for edge in tokens[1:-3]:
                    # middle edges
                    src, tgt, _  = edge.split(",")
                    e_src = int(src[2:]) - 1
                    e_tgt = int(tgt[1:-1]) - 1

                    e2p[e_src, e_tgt, p] = 1
                    edges.append((e_src, e_tgt))

                    p2e[p, e_src, e_tgt] = 1
                requests[-1].append(edges)
                p += 1
            else:
                raise ValueError
        return requests, r2p, r2p_tuple, P, Pr, e2p, p2e

def load_problem():
    demands = process_demands(demand_file)
    constraints = process_topology(topology_file)
    paths, r2p, r2p_tup, P, Pr, e2p, p2e = process_paths(path_file, demands)
    graph = nx.Graph(constraints)

    return Problem(
        graph,
        paths,
        demands,
        constraints,
        P,
        Pr,
        e2p,
        p2e,
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

V = problem.constraints.shape[0]

# construct dense_r2p as a sequence of [ 0 ... 1 ... 0 ] stacked vectors
dense_r2p = []
for v in range(V * V):
    vec = np.zeros(problem.P)
    vec[v*problem.Pr:(1+v)*problem.Pr] = 1
    dense_r2p.append(vec)
dense_r2p = np.vstack(dense_r2p)

# CVXPY
X = cp.Variable(problem.P, nonneg=True)
objective = cp.Minimize(-cp.sum(X))
constraints = [
    dense_r2p @ X <= problem.demand.reshape((V*V,)),
    problem.e2p.reshape((V*V, problem.P)) @ X <= problem.constraints.reshape((V*V,)),
]

prob = cp.Problem(objective, constraints)
result = prob.solve(verbose=True)
#print(X.value)
#print(constraints[0].dual_value)
print(prob.value)

sol = X.value

print("Constraint residuals")
print((dense_r2p @ sol - problem.demand.reshape((V*V,))).max())
print((problem.e2p.reshape((V*V, problem.P)) @ sol - problem.constraints.reshape((V*V,))).max())

# ADMM

def naive_admm_solver(
    problem,
    rho,
    x, z, s1, s3,
    lambda1, lambda3, lambda4,
    num_iters = 100,
):
    Pr = problem.Pr

    d = problem.demand.reshape(-1)
    c = problem.constraints.reshape(-1)

    e2p = problem.e2p.reshape(V*V, -1)

    e2p_flat = e2p.astype(int)
    num_paths_for_edge = e2p_flat.sum(-1)
    num_edges_for_path = e2p_flat.sum(0)
    # clamp to 1?
    num_edges_for_path = np.maximum(1, num_edges_for_path)

    e2p_flat_bool = e2p.reshape(-1).astype(bool)

    # A_r is wrong
    ones = np.ones((Pr, Pr))
    eye = np.eye(Pr)[None] * num_edges_for_path.reshape((V*V, Pr))[:, None]
    A_r_inv = np.linalg.inv(ones[None] + eye)

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
        b = (
            (-1 - lambda1[:,None] + (e2p * lambda4).sum(0).reshape(V*V, -1))
            + rho * (-d + s1 + (e2p * z).sum(1))[:,None]
        )
        x = -np.einsum("nab,nb->na", A_r_inv / rho, b).reshape(-1)
        x = np.maximum(0, x)

        # z update
        for e in range(c.size):
            if A_invs[e] is not None:
                A_inv = A_invs[e] / rho
                path_ix = e2p[e].astype(bool)
                #b = (-lambda1[e,None] - lambda2[e, path_ix] + lambda5[e, path_ix]
                b = (-lambda3[e,None] - lambda4[e, path_ix]
                    + rho * (-c[e,None] + s3[e,None] - x[path_ix]))
                z[e, path_ix] = -A_inv @ b
        z = np.maximum(0, z)
        #import pdb; pdb.set_trace()
         
        # s update
        s1 = (lambda1 + rho * (d - x.reshape(V*V, -1).sum(1))) / rho
        s3 = (lambda3 + rho * (c - (e2p * z).sum(-1))) / rho
        s1 = np.maximum(0, s1)
        s3 = np.maximum(0, s3)

        r1 = d - x.reshape(V*V, -1).sum(1) - s1
        r3 = c - (e2p * z).sum(-1) - s3
        r4 = x[None] - z

        # lambda update
        lambda1 += rho * r1
        lambda3 += rho * r3
        lambda4 += rho * r4

        lambda4 = lambda4.reshape(-1)
        lambda4[e2p_flat_bool] += rho * r4.reshape(-1)[e2p_flat_bool]
        lambda4 = lambda4.reshape(e2p_flat.shape)

    return x, z, s1, s3, lambda1, lambda3, lambda4, r1, r3, r4


x, z, s1, s3, l1, l3, l4, r1, r3, r4 = naive_admm_solver(
    problem,
    rho = 1,
    x = np.zeros((problem.P,)),
    z = np.zeros((V*V, problem.P)),
    s1 = np.zeros((V*V,)),
    s3 = np.zeros((V*V,)),
    lambda1 = np.zeros((V*V,)),
    lambda3 = np.zeros((V*V,)),
    lambda4 = np.zeros((V*V, problem.P)),
    num_iters = 10000,
)
#print(s1)
#print(s3)
print(-x.sum())

import pdb; pdb.set_trace()
