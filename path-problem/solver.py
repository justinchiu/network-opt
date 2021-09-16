import numpy as np

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
            #if k > 0 else None
            for k in num_paths_for_edge.tolist()
        ]

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

    b = (
        (-1 - lambda1[:,None] + (e2p * lambda4).sum(0).reshape(V*V, -1))
        + rho * ((-d + s1)[:,None] - (e2p * z).sum(0).reshape(V*V, -1))
    )
    x = -np.einsum("nab,nb->na", A_r_inv / rho, b).reshape(-1)
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

    A_invs = cache.A_invs
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

    return r1, r3, r4

def update_lambda(variables, residuals, cache):
    lambda1 = variables.l1
    lambda3 = variables.l3
    lambda4 = variables.l4

    rho = cache.rho

    r1, r3, r4 = residuals

    return (lambda1 + rho * r1, lambda3 + rho * r3, lambda4 + rho * r4)


def update_x_cvxpy(variables, constraints, cache):
    z = variables.z
    s1 = variables.s1
    lambda1 = variables.l1
    lambda4 = variables.l4

    d = constraints.d

    e2p = cache.e2p
    A_r_inv = cache.A_r_inv
    V = cache.V
    rho = cache.rho

    b = (
        (-1 - lambda1[:,None] + (e2p * lambda4).sum(0).reshape(V*V, -1))
        + rho * ((-d + s1)[:,None] - (e2p * z).sum(0).reshape(V*V, -1))
    )
    x = -np.einsum("nab,nb->na", A_r_inv / rho, b).reshape(-1)
    return np.maximum(0, x)
