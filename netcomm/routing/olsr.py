import heapq
from typing import Dict, List

import numpy as np


class OLSRPolicy:
    uses_regime = False

    def __init__(self, cfg, ema_alpha: float = 0.3):
        # why: OLSR is proactive — it keeps a smoothed link-state estimate that
        # lags the truth. We use an exponential moving average of observed
        # pi_up as the routable link-state.
        self.cfg = cfg
        self.alpha = float(ema_alpha)
        self._estimate: np.ndarray = None  # type: ignore[assignment]

    def _update_estimate(self, pi_up_np: np.ndarray):
        if self._estimate is None or self._estimate.shape != pi_up_np.shape:
            self._estimate = pi_up_np.copy()
            return
        self._estimate = (1.0 - self.alpha) * self._estimate + self.alpha * pi_up_np

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
        pi_np = np.asarray(pi_up)
        self._update_estimate(pi_np)
        edge_cost = -np.log(np.clip(self._estimate, 1e-9, 1.0))
        adj_np = np.asarray(adj)
        return [self._dijkstra(int(s), int(d), adj_np, edge_cost)
                for (s, d) in flows]
