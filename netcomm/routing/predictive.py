import heapq
import numpy as np


def predictive_forward_step(i, dest, neighborhood, pi_up_row, positions, p_min):
    if len(neighborhood) == 0:
        return -1
    pos_d = positions[dest]
    dist_i = np.linalg.norm(positions[i] - pos_d)
    dist_n = np.linalg.norm(positions[neighborhood] - pos_d, axis=-1)
    geo_progress = np.clip(dist_i - dist_n, 0.0, None) / (dist_i + 1e-6)
    pi_neighbors = pi_up_row[neighborhood]
    score = np.where(pi_neighbors > p_min, pi_neighbors * geo_progress, -1.0)
    best = int(np.argmax(score))
    if score[best] <= 0.0:
        return -1
    return int(neighborhood[best])


def predictive_route(src, dest, positions, adj, pi_up, p_min, max_hops=16):
    return predictive_route_bfs(src, dest, positions, adj, pi_up, p_min, max_hops)


def predictive_route_greedy(src, dest, positions, adj, pi_up, p_min, max_hops=16):
    path = [src]
    cur = src
    visited = {src}
    for _ in range(max_hops):
        if cur == dest:
            return path
        nbrs = np.where(adj[cur])[0]
        nbrs = np.array([n for n in nbrs if n not in visited], dtype=np.int32)
        nxt = predictive_forward_step(cur, dest, nbrs, pi_up[cur], positions, p_min)
        if nxt < 0:
            return []
        path.append(int(nxt))
        visited.add(int(nxt))
        cur = int(nxt)
    return [] if cur != dest else path


def predictive_route_bfs(src, dest, positions, adj, pi_up, p_min, max_hops=16):
    if src == dest:
        return [src]
    n = adj.shape[0]
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
            if not adj[u, v]:
                continue
            p = float(pi_up[u, v])
            if p < p_min:
                continue
            nc = c - np.log(max(p, 1e-9))
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
