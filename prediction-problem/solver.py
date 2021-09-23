import numpy as np
import cvxpy as cp

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

class Variables:
    def __init__(self, V, Pr):
        self.x  = np.zeros((V*V*Pr,))
        self.z  = np.zeros((V*V, V*V*Pr,))
        self.s1 = np.zeros((V*V,))
        self.s3 = np.zeros((V*V,))
        self.l1 = np.zeros((V*V,))
        self.l3 = np.zeros((V*V,))
        self.l4 = np.zeros((V*V, V*V*Pr))

class Constraints:
    def __init__(self, d, c):
        self.d = d
        self.c = c

class Cache:
    def __init__(self, V, Pr, rho, e2p):
        self.V = V
        self.Pr = Pr
        self.rho = rho

        self.e2p = e2p.reshape(V*V, -1)

        e2p_flat = self.e2p.astype(int)
        num_paths_for_edge = e2p_flat.sum(-1)
        num_edges_for_path = e2p_flat.sum(0)

        e2p_flat_bool = e2p.reshape(-1).astype(bool)

        ones = np.ones((Pr, Pr))
        eye = np.eye(Pr)[None] * num_edges_for_path.reshape((V*V, Pr))[:, None]
        self.A_r_inv = np.linalg.inv(ones[None] + eye)

        self.A_invs = [
            np.linalg.inv(np.ones((k, k)) + np.eye(k))
            for k in num_paths_for_edge.tolist()
        ]

        # DBG
        # construct dense_r2p as a sequence of [ 0 ... 1 ... 0 ] stacked vectors
        dense_r2p = []
        for v in range(V * V):
            vec = np.zeros(V*V*Pr)
            vec[v*Pr:(1+v)*Pr] = 1
            dense_r2p.append(vec)
        self.dense_r2p = np.vstack(dense_r2p)

class StatLog:
    def __init__(self):
        self.objective = []
        self.lagrange_multipliers = []
        self.r1 = []
        self.r3 = []
        self.r4 = []

    def log_stats(self, variables, residuals):
        self.objective.append(variables.x.sum())

        r1, r3, r4 = residuals
        self.r1.append(np.square(r1).sum())
        self.r3.append(np.square(r3).sum())
        self.r4.append(np.square(r4).sum())

    def print(self):
        L = len(self.objective)
        kv_map = {
            "iter": range(L),
            "objective": self.objective,
            "r1": self.r1,
            "r3": self.r3,
            "r4": self.r4,
        }
        df = pd.DataFrame(kv_map)

        fig, axes = plt.subplots(len(kv_map)-1, figsize=(18, 10))
        for i, k in enumerate(k for k in kv_map if k != "iter"):
            sns.lineplot(ax=axes[i], data=df, x="iter", y=k)

        plt.show()

def update_x(variables, constraints, cache):
    z = variables.z
    s1 = variables.s1
    lambda1 = variables.l1
    lambda4 = variables.l4

    d = constraints.d

    e2p = cache.e2p
    A_r_inv = cache.A_r_inv
    V = cache.V
    rho = cache.rho
    Pr = cache.Pr

    b = (
        (-1 - lambda1[:,None] + (e2p * lambda4).sum(0).reshape(V*V, Pr))
        + rho * ((-d + s1)[:,None] - (e2p * z).sum(0).reshape(V*V, Pr))
    )
    x = -np.einsum("nab,nb->na", A_r_inv / rho, b).reshape(-1)
    #import pdb; pdb.set_trace()
    return np.maximum(0, x)

def update_z(variables, constraints, cache):
    x = variables.x
    s3 = variables.s3
    lambda3 = variables.l3
    lambda4 = variables.l4

    c = constraints.c

    A_invs = cache.A_invs
    rho = cache.rho
    e2p = cache.e2p

    z = np.zeros(variables.z.shape)
    for e in range(c.size):
        if A_invs[e] is not None:
            A_inv = A_invs[e] / rho
            path_ix = e2p[e].astype(bool)
            b = (-lambda3[e,None] - lambda4[e, path_ix]
                + rho * (-c[e,None] + s3[e,None] - x[path_ix]))
            z[e, path_ix] = -A_inv @ b
    return np.maximum(0, z)

def update_s(variables, constraints, cache):
    x = variables.x
    z = variables.z
    s3 = variables.s3
    lambda1 = variables.l1
    lambda3 = variables.l3
    lambda4 = variables.l4

    d = constraints.d
    c = constraints.c

    e2p = cache.e2p
    V = cache.V
    rho = cache.rho

    s1 = (lambda1 + rho * (d - x.reshape(V*V, -1).sum(1))) / rho
    s3 = (lambda3 + rho * (c - (e2p * z).sum(-1))) / rho
    s1 = np.maximum(0, s1)
    s3 = np.maximum(0, s3)
    return s1, s3

def compute_residuals(variables, constraints, cache):
    x = variables.x
    z = variables.z
    s1 = variables.s1
    s3 = variables.s3

    d = constraints.d
    c = constraints.c

    V = cache.V
    e2p = cache.e2p

    r1 = d - x.reshape(V*V, -1).sum(1) - s1
    r3 = c - (e2p * z).sum(-1) - s3
    r4 = x[None] - z
    r4[~e2p.astype(bool)] = 0

    return r1, r3, r4

def update_lambda(variables, residuals, cache):
    lambda1 = variables.l1
    lambda3 = variables.l3
    lambda4 = variables.l4

    rho = cache.rho

    r1, r3, r4 = residuals

    return (lambda1 + rho * r1, lambda3 + rho * r3, lambda4 + rho * r4)


## TESTING

def update_x_cvxpy(variables, constraints, cache):
    z = variables.z
    s1 = variables.s1
    lambda1 = variables.l1
    lambda4 = variables.l4

    d = constraints.d

    e2p = cache.e2p
    A_r_inv = cache.A_r_inv
    V = cache.V
    Pr = cache.Pr
    rho = cache.rho
    dense_r2p = cache.dense_r2p

    x = cp.Variable(V*V*Pr, nonneg=True)
    objective = cp.Minimize(
        -cp.sum(x) + cp.sum(cp.multiply(lambda1, (d - dense_r2p @ x - s1)))
        + cp.sum(cp.multiply(lambda4, x[None] - z))
        + (rho / 2) * (
            cp.sum((d - dense_r2p @ x - s1)**2)
            + cp.sum((x[None] - z)[e2p.astype(bool)]**2)
        )
    )
    prob = cp.Problem(objective)
    result = prob.solve()
    return x.value

def update_z_cvxpy(variables, constraints, cache):
    x = variables.x
    s3 = variables.s3
    lambda3 = variables.l3
    lambda4 = variables.l4

    c = constraints.c

    A_invs = cache.A_invs
    rho = cache.rho
    e2p = cache.e2p
    V = cache.V
    Pr = cache.Pr

    z = cp.Variable((V*V, V*V*Pr), nonneg=True)
    objective = cp.Minimize(
        cp.sum(cp.multiply(lambda3, c - cp.sum(cp.multiply(e2p, z), axis=1) - s3))
        + cp.sum(cp.multiply(lambda4, x[None] - z))
        + (rho / 2) * (
            cp.sum((c - cp.sum(cp.multiply(e2p, z), axis=1) - s3)**2)
            + cp.sum((x[None] - z)[e2p.astype(bool)]**2)
        )
    )
    prob = cp.Problem(objective)
    result = prob.solve()
    return z.value

def update_s_cvxpy(variables, constraints, cache):
    x = variables.x
    z = variables.z
    lambda1 = variables.l1
    lambda3 = variables.l3

    d = constraints.d
    c = constraints.c

    e2p = cache.e2p
    V = cache.V
    Pr = cache.Pr
    rho = cache.rho
    dense_r2p = cache.dense_r2p

    s1 = cp.Variable(V*V, nonneg=True)
    objective = cp.Minimize(
        cp.sum(cp.multiply(lambda1, (d - dense_r2p @ x - s1)))
        + (rho / 2) * cp.sum((d - dense_r2p @ x - s1)**2)
    )
    prob = cp.Problem(objective)
    result = prob.solve()

    s3 = cp.Variable(V*V, nonneg=True)
    objective = cp.Minimize(
        cp.sum(cp.multiply(lambda3, c - cp.sum(cp.multiply(e2p, z), axis=1) - s3))
        + (rho / 2) * cp.sum((c - cp.sum(cp.multiply(e2p, z), axis=1) - s3)**2)
    )
    prob = cp.Problem(objective)
    result = prob.solve()
    return s1.value, s3.value


if __name__ == "__main__":
    from data import load_problem

    problem = load_problem()

    rho = 1
    Pr = problem.Pr
    V = problem.constraints.shape[0]

    d = problem.demand.reshape(-1)
    c = problem.constraints.reshape(-1)

    # see solver.py
    variables = Variables(V, Pr)
    constraints = Constraints(d, c)
    cache = Cache(V, Pr, rho, problem.e2p)

    x0 = update_x_cvxpy(variables, constraints, cache)
    x = update_x(variables, constraints, cache)
    assert(np.allclose(x, x0))
    variables.x = x

    z0 = update_z_cvxpy(variables, constraints, cache)
    z = update_z(variables, constraints, cache)
    assert np.allclose(z, z0)
    variables.z = z

    s1, s3 = update_s(variables, constraints, cache)
    s10, s30 = update_s_cvxpy(variables, constraints, cache)
    assert np.allclose(s1, s10)
    assert np.allclose(s3, s30)
    variables.s1 = s1
    variables.s3 = s3

    residuals = compute_residuals(variables, constraints, cache)

    l1, l3, l4 = update_lambda(variables, residuals, cache)
