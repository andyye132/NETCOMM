import heapq
import numpy as np


def learning_route(src, dest, positions, adj, pi_up, sinr, gamma_th,
                   alpha=1.0, beta=0.5, gamma=0.1, max_hops=16):
    # RL-inspired weighted-Dijkstra approximation (trained policy is future work).
    # Edge score q(i,j) = alpha*log(pi_up) + beta*log(sinr/gamma_th)
    #                   - gamma*hops_lower_bound_to_dest.
    if src == dest:
        return [src]
    n = adj.shape[0]
    pos_d = positions[dest]
    # Hop-count lower bound proxy: geographic distance / max single-hop range.
    dist_to_d = np.linalg.norm(positions - pos_d, axis=-1)
    rng = float(np.max(dist_to_d)) + 1e-6
    hops_lb = dist_to_d / rng  # normalized [0, 1]
    INF = float("inf")
    cost = np.full(n, INF, dtype=np.float64)
    cost[src] = 0.0
    prev = -np.ones(n, dtype=np.int32)
    depth = np.zeros(n, dtype=np.int32)
    pq = [(0.0, int(src))]
    while pq:
        c, u = heapq.heappop(pq)
        if c > cost[u]:
            continue
        if u == dest:
            break
        if depth[u] >= max_hops:
            continue
        for v in range(n):
            if not adj[u, v] or v == u:
                continue
            p = float(pi_up[u, v])
            s = float(sinr[u, v])
            if p <= 0.0 or s <= 0.0:
                continue
            q = (alpha * np.log(max(p, 1e-9))
                 + beta * np.log(max(s / max(gamma_th, 1e-9), 1e-9))
                 - gamma * float(hops_lb[v]))
            nc = c + (-q)
            if nc < cost[v]:
                cost[v] = nc
                prev[v] = u
                depth[v] = depth[u] + 1
                heapq.heappush(pq, (nc, int(v)))
    if cost[dest] == INF:
        return []
    path = [int(dest)]
    cur = int(prev[dest])
    while cur >= 0:
        path.append(cur)
        cur = int(prev[cur])
    return list(reversed(path))
