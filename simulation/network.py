import numpy as np
import networkx as nx

class Network():
    def __init__(self, N: int, k_ave: float, edge_list: np.ndarray[int], address_list: np.ndarray[int], cursor_list: np.ndarray[int]):
        self.N = N
        self.k_ave = k_ave
        self.edge_list = edge_list
        self.address_list = address_list
        self.cursor_list = cursor_list

def ER(N: int, p: float) -> Network:
    g = nx.erdos_renyi_graph(N, p)
    k_ave = np.mean([len(list(g.neighbors(node))) for node in g.nodes])
    edge_list = np.zeros(2*g.number_of_edges(), dtype=int)
    address_list = np.zeros(N, dtype=int)
    cursor_list = np.zeros(N, dtype=int)

    for i in range(N-1):
        node = list(g.nodes)[i]
        address_list[node + 1] = address_list[node] + len(list(g.neighbors(node)))
        cursor_list[node + 1] = address_list[node + 1]
    
    for edge in list(g.edges):
        edge_list[cursor_list[edge[0]]] = edge[1]
        edge_list[cursor_list[edge[1]]] = edge[0]
        cursor_list[edge[0]] += 1
        cursor_list[edge[1]] += 1

    return Network(g.number_of_nodes(), k_ave, edge_list, address_list, cursor_list)

def BA(N: int, k_ave: float) -> Network:
    g = nx.barabasi_albert_graph(N, int(k_ave))
    k_ave = np.mean([len(list(g.neighbors(node))) for node in g.nodes])
    edge_list = np.zeros(2*g.number_of_edges(), dtype=int)
    address_list = np.zeros(N, dtype=int)
    cursor_list = np.zeros(N, dtype=int)

    for i in range(N-1):
        node = list(g.nodes)[i]
        address_list[node + 1] = address_list[node] + len(list(g.neighbors(node)))
        cursor_list[node + 1] = address_list[node + 1]
    
    for edge in list(g.edges):
        edge_list[cursor_list[edge[0]]] = edge[1]
        edge_list[cursor_list[edge[1]]] = edge[0]
        cursor_list[edge[0]] += 1
        cursor_list[edge[1]] += 1

    return Network(g.number_of_nodes(), k_ave, edge_list, address_list, cursor_list)