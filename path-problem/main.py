from pathlib import Path

from itertools import product

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

from typing import NamedTuple, List

from solver import (
    Variables, Constraints, Cache,
    update_x, update_z, update_s,
    compute_residuals, update_lambda,
)

#from solver import (
    #update_x_cvxpy, update_z_cvxpy, update_s_cvxpy,
#)

class Problem(NamedTuple):
    graph: nx.Graph
    paths: List[List[int]]
    demand: np.ndarray # in adjacency matrix form
    constraints: np.ndarray # in adjacency matrix form
    P: int
    Pr: int
    e2p: np.ndarray # binary map from edge to which paths appear


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
                requests[-1].append(edges)
                p += 1
            else:
                raise ValueError
        # for self-loops add a self-path for self-routes
        for v in range(V):
            e2p.reshape((V,V,V,V,-1))[v,v,v,v] = 1
        return requests, r2p, r2p_tuple, P, Pr, e2p 

def load_problem():
    demands = process_demands(demand_file)
    constraints = process_topology(topology_file)
    paths, r2p, r2p_tup, P, Pr, e2p = process_paths(path_file, demands)
    graph = nx.Graph(constraints)

    return Problem(
        graph,
        paths,
        demands,
        constraints,
        P,
        Pr,
        e2p,
    )


problem = load_problem()

G = problem.graph
pos = nx.spring_layout(G, seed=225)  # Seed for reproducible layout
nx.draw(G, pos, with_labels=True)

plt.savefig('example_graph.png')

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
    # reshaping does not work with cvxpy
    #cp.sum(cp.reshape(X, (V*V, 6)), axis=1) <= problem.demand.reshape((V*V,)),
    problem.e2p.reshape((V*V, problem.P)) @ X <= problem.constraints.reshape((V*V,)),
]

prob = cp.Problem(objective, constraints)
result = prob.solve(verbose=True)
print(prob.value)

sol = X.value

print("Constraint residuals")
print((dense_r2p @ sol - problem.demand.reshape((V*V,))).max())
print((problem.e2p.reshape((V*V, problem.P)) @ sol - problem.constraints.reshape((V*V,))).max())

# ADMM

def naive_admm_solver(
    problem,
    rho,
    num_iters = 100,
):
    Pr = problem.Pr
    V = problem.constraints.shape[0]

    d = problem.demand.reshape(-1)
    c = problem.constraints.reshape(-1)

    # see solver.py
    variables = Variables(V, Pr)
    constraints = Constraints(d, c)
    cache = Cache(V, Pr, rho, problem.e2p)

    for iter in range(num_iters):
        x = update_x(variables, constraints, cache)
        variables.x = x
        z = update_z(variables, constraints, cache)
        variables.z = z
        s1, s3 = update_s(variables, constraints, cache)
        variables.s1 = s1
        variables.s3 = s3

        residuals = compute_residuals(variables, constraints, cache)

        l1, l3, l4 = update_lambda(variables, residuals, cache)
        variables.l1 = l1
        variables.l3 = l3
        variables.l4 = l4

    return variables, residuals


variables, (r1, r3, r4)= naive_admm_solver(
    problem,
    rho = 1,
    num_iters = 1000,
)
print(-variables.x.sum())

print(np.square(r4).sum())
import pdb; pdb.set_trace()
