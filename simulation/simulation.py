from .network import Network
import numpy as np

def simulate_sir(graph: Network, t_max: int, initial_infected: np.ndarray[int], lamb: float, gamma: float, initial_recovered: np.ndarray[int] = np.zeros(0, dtype=int)):
    N = graph.N

    num_s = np.zeros(t_max + 1, dtype=int)
    num_i = np.zeros(t_max + 1, dtype=int)
    num_r = np.zeros(t_max + 1, dtype=int)

    to_infect = set()
    to_recover = set()

    infected = set(initial_infected)

    state = np.zeros(N, dtype=int)

    for node in initial_infected:
        state[node] = 1
    for node in initial_recovered:
        state[node] = 2

    num_i[0] = len(state[state == 1])
    num_r[0] = len(state[state == 2])
    num_s[0] = N - num_i[0] - num_r[0]

    t = 0
    while t < t_max and num_i[t] > 0:
        for infected_node in infected:
            deg = graph.cursorList[infected_node] - graph.addressList[infected_node]
            for neighbor_idx in range(deg):
                neighbor = graph.edge_list[graph.addressList[infected_node] + neighbor_idx]
                if state[neighbor] == 0:
                    if np.random.rand() < lamb:
                        to_infect.add(neighbor)

            if np.random.rand() < gamma:
                to_recover.add(infected_node)

        for node in to_infect:
            infected.add(node)
            state[node] = 1
            num_i[t + 1] += 1
            num_s[t + 1] -= 1

        for node in to_recover:
            infected.remove(node)
            state[node] = 2
            num_i[t + 1] -= 1
            num_r[t + 1] += 1

        t += 1
        to_infect.clear()
        to_recover.clear()
    
    if t != t_max:
        num_s[t + 1:] = num_s[t]
        num_i[t + 1:] = num_i[t]
        num_r[t + 1:] = num_r[t]

    return num_s, num_i, num_r
