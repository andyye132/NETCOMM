import heapq
from typing import List

import numpy as np


def _dijkstra(adj: np.ndarray, src: int, dst: int, edge_cost: np.ndarray,
              blocked_edges: set, blocked_nodes: set) -> List[int]:
    n = adj.shape[0]
    INF = float("inf")
    cost = np.full(n, INF)
    cost[src] = 0.0
    prev = -np.ones(n, dtype=np.int32)
    pq = [(0.0, int(src))]
    while pq:
        c, u = heapq.heappop(pq)
        if c > cost[u]:
            continue
        if u == dst:
            break
        for v in range(n):
            if not adj[u, v]:
                continue
            if v in blocked_nodes and v != dst:
                continue
            if (u, v) in blocked_edges:
                continue
            w = float(edge_cost[u, v])
            if not np.isfinite(w):
                continue
            nc = c + w
            if nc < cost[v]:
                cost[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, int(v)))
    if cost[dst] == INF:
        return []
    path = [int(dst)]
    cur = int(prev[dst])
    while cur >= 0:
        path.append(cur)
        cur = int(prev[cur])
    return list(reversed(path))


def k_disjoint_paths(adj: np.ndarray, src: int, dst: int, k: int,
                     edge_cost: np.ndarray) -> List[List[int]]:
    # why: Yen-style k-shortest with node-disjoint enforcement. For each
    # already-found path, we ban all its intermediate nodes; this is the simple
    # "node-disjoint" variant the paper calls for (k partially independent
    # paths). If fewer than k node-disjoint paths exist we return what we have.
    paths: List[List[int]] = []
    blocked_nodes: set = set()
    blocked_edges: set = set()
    for _ in range(k):
        p = _dijkstra(adj, src, dst, edge_cost, blocked_edges, blocked_nodes)
        if not p:
            break
        paths.append(p)
        for v in p[1:-1]:
            blocked_nodes.add(int(v))
    return paths
