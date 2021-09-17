
from pathlib import Path

import networkx as nx
import numpy as np

from typing import NamedTuple, List

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
