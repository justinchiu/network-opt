from pathlib import Path

from itertools import product

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

from typing import NamedTuple, List

from data import load_problem

from solver import (
    StatLog,
    Variables, Constraints, Cache,
    update_x, update_z, update_s,
    compute_residuals, update_lambda,
)

from solver import (
    update_x_cvxpy, update_z_cvxpy, update_s_cvxpy,
)


problem = load_problem()

"""
G = problem.graph
pos = nx.spring_layout(G, seed=225)  # Seed for reproducible layout
nx.draw(G, pos, with_labels=True)

plt.savefig('example_graph.png')
"""

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
print((problem.e2p.reshape((V*V, problem.P)) @ sol
    - problem.constraints.reshape((V*V,))).max())

# ADMM

def naive_admm_solver(
    problem,
    rho,
    num_iters = 100,
):
    stat_log = StatLog()

    Pr = problem.Pr
    V = problem.constraints.shape[0]

    d = problem.demand.reshape(-1)
    c = problem.constraints.reshape(-1)

    # see solver.py
    variables = Variables(V, Pr)
    constraints = Constraints(d, c)
    cache = Cache(V, Pr, rho, problem.e2p)
    # DEBUG
    cache.dense_r2p = dense_r2p

    for iter in range(num_iters):
        #x = update_x(variables, constraints, cache)
        x = update_x_cvxpy(variables, constraints, cache)
        variables.x = x
        #z = update_z(variables, constraints, cache)
        z = update_z_cvxpy(variables, constraints, cache)
        variables.z = z
        #s1, s3 = update_s(variables, constraints, cache)
        s1, s3 = update_s_cvxpy(variables, constraints, cache)
        variables.s1 = s1
        variables.s3 = s3

        residuals = compute_residuals(variables, constraints, cache)

        l1, l3, l4 = update_lambda(variables, residuals, cache)
        variables.l1 = l1
        variables.l3 = l3
        variables.l4 = l4

        stat_log.log_stats(variables, residuals)

    return variables, residuals, stat_log


variables, (r1, r3, r4), stat_log = naive_admm_solver(
    problem,
    rho = 1,
    num_iters = 50,
)
print(-variables.x.sum())

print(np.square(r4).sum())

stat_log.print()

#import pdb; pdb.set_trace()
