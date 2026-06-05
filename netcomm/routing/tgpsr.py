import numpy as np


class TGPSRPolicy:
    uses_regime = False

    def __init__(self, cfg, lookahead_default: float = 0.2):
        # why: trajectory-aware GPSR — greedy forward to the neighbor whose
        # predicted position at t + Delta_p / n_hops_estimate is closest to
        # destination. We use a fixed n_hops_estimate of 3 to avoid recursion;
        # in practice this is set per-flow by a separate estimator.
        self.cfg = cfg
        self.lookahead_default = float(lookahead_default)

    def _route_one(self, src: int, dst: int, positions, velocities,
                   adj, sinr, lookahead: float, max_hops: int = 16):
        if src == dst:
            return [src]
        pred = positions + velocities * lookahead
        path = [src]
        cur = src
        visited = {src}
        for _ in range(max_hops):
            if cur == dst:
                return path
            nbrs = np.where(adj[cur])[0]
            nbrs = np.array([n for n in nbrs if n not in visited], dtype=np.int32)
            if len(nbrs) == 0:
                return []
            gate = sinr[cur, nbrs] > self.cfg.gamma_th
            if not np.any(gate):
                return []
            dist = np.linalg.norm(pred[nbrs] - pred[dst], axis=-1)
            score = np.where(gate, -dist, -np.inf)
            best = int(np.argmax(score))
            if not np.isfinite(score[best]):
                return []
            cur = int(nbrs[best])
            path.append(cur)
            visited.add(cur)
        return path if path[-1] == dst else []

    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        pos_np = np.asarray(positions)
        vel_np = (np.asarray(node_state.vel) if node_state is not None
                  else np.zeros_like(pos_np))
        sinr_np = np.asarray(sinr)
        adj_np = np.asarray(adj)
        return [self._route_one(int(s), int(d), pos_np, vel_np, adj_np,
                                sinr_np, self.lookahead_default)
                for (s, d) in flows]
