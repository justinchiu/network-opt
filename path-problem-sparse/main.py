import pickle
import json
import torch
import numpy as np

from networkx.readwrite import json_graph

from ADMM import ADMM


def get_regular_path(path_dict, num_path):
    """Return path dictionary with the same number of paths per demand.
    Fill with the first path when number of paths is not enough.
    """

    for (s_k, t_k) in path_dict:
        if len(path_dict[(s_k, t_k)]) < num_path:
            path_dict[(s_k, t_k)] = [
                path_dict[(s_k, t_k)][0] for _
                in range(num_path - len(path_dict[(s_k, t_k)]))]\
                + path_dict[(s_k, t_k)]
        elif len(path_dict[(s_k, t_k)]) > num_path:
            path_dict[(s_k, t_k)] = path_dict[(s_k, t_k)][:num_path]
    return path_dict


def get_topo_matrix(path_dict, num_path):
    """
    Return matrices related to topology.
    p2e: [path_node_idx, edge_nodes_inx]
    """

    # get regular path dict
    path_dict = get_regular_path(path_dict, num_path)

    # edge nodes' degree, index lookup
    edge2idx_dict = {edge: idx for idx, edge in enumerate(G.edges)}
    node2degree_dict = {}
    edge_num = len(G.edges)

    # build edge_index
    src, dst, path_i = [], [], 0
    for s in range(len(G)):
        for t in range(len(G)):
            if s == t:
                continue
            for path in path_dict[(s, t)]:
                for (u, v) in zip(path[:-1], path[1:]):
                    src.append(edge_num+path_i)
                    dst.append(edge2idx_dict[(u, v)])

                    if src[-1] not in node2degree_dict:
                        node2degree_dict[src[-1]] = 0
                    node2degree_dict[src[-1]] += 1
                    if dst[-1] not in node2degree_dict:
                        node2degree_dict[dst[-1]] = 0
                    node2degree_dict[dst[-1]] += 1
                path_i += 1

    p2e = torch.tensor([src, dst], dtype=torch.long)
    p2e[0] -= len(G.edges)

    return p2e


if __name__ == '__main__':

    # ============== initializaiton ==============
    # load device, number of path, traffic demand matrix, path, topology, p2e
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    num_path = 4
    tm = pickle.load(open('data/UsCarrier_traffic-matrix.pkl', 'rb'))
    pickle.dump(tm, open('data/UsCarrier_traffic-matrix.pkl', 'wb'))
    path_dict = pickle.load(open('data/UsCarrier_path-dict.pkl', 'rb'))
    G = json_graph.node_link_graph(json.load(open('data/UsCarrier.json')))
    p2e = get_topo_matrix(path_dict, num_path).to(device)

    # init admm
    admm = ADMM(
        p2e=p2e, 
        num_path=num_path, 
        num_path_node=num_path*G.number_of_nodes()*(G.number_of_nodes()-1),
        num_edge_node=G.number_of_edges(), 
        rho=1, 
        device=device)

    # ============== ADMM steps ==============
    # observation = edge capacity + traffic matrix duplicated by number of path
    tm = torch.FloatTensor(
        [[ele]*num_path for i, ele in enumerate(tm.flatten())
            if i % len(tm) != i//len(tm)]).flatten().to(device)
    capacity = torch.FloatTensor(
        [float(c_e) for u, v, c_e in G.edges.data('capacity')]).to(device)
    obs = torch.concat([capacity, tm])

    # =================================================================
    # === Feel free to change below for different starting solution ===
    # =================================================================
    # action: 10% of corresponding demand for each edge
    action = tm.clone() * 0.1
    # =================================================================

    # use admm to fine tune the action
    admm.tune_action(
        obs=obs, 
        action=action, 
        num_admm_step=50)
    admm.stat_log.print()

    