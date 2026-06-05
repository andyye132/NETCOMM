import heapq
from typing import List

import numpy as np


class POLSRPolicy:
    uses_regime = False

    def __init__(self, cfg):
        # why: predictive OLSR variant — like OLSR but the link-state is the
        # LCB-of-survival rather than smoothed pi_up. No EMA: each epoch the
        # LCB is recomputed from the current forecast.
        self.cfg = cfg

    def _dijkstra(self, src: int, dst: int, adj: np.ndarray,
                  edge_cost: np.ndarray, max_hops: int = 32) -> List[int]:
        n = adj.shape[0]
        INF = float("inf")
        cost = np.full(n, INF)
        cost[src] = 0.0
        prev = -np.ones(n, dtype=np.int32)
        depth = np.zeros(n, dtype=np.int32)
        pq = [(0.0, int(src))]
        while pq:
            c, u = heapq.heappop(pq)
            if c > cost[u]:
                continue
            if u == dst:
                break
            if depth[u] >= max_hops:
                continue
            for v in range(n):
                if not adj[u, v] or v == u:
                    continue
                w = float(edge_cost[u, v])
                if not np.isfinite(w):
                    continue
                nc = c + w
                if nc < cost[v]:
                    cost[v] = nc
                    prev[v] = u
                    depth[v] = depth[u] + 1
                    heapq.heappush(pq, (nc, int(v)))
        if cost[dst] == INF:
            return []
        path = [int(dst)]
        cur = int(prev[dst])
        while cur >= 0:
            path.append(cur)
            cur = int(prev[cur])
        return list(reversed(path))

    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        adj_np = np.asarray(adj)
        if lcb is not None:
            link_state = np.asarray(lcb.lcb)
        else:
            link_state = np.asarray(pi_up)
        edge_cost = -np.log(np.clip(link_state, 1e-9, 1.0))
        return [self._dijkstra(int(s), int(d), adj_np, edge_cost)
                for (s, d) in flows]
