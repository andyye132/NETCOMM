import numpy as np


def p3_route(src, dest, positions, velocities, adj, pi_up, p_min,
             T_lookahead=0.5, max_hops=16):
    # P^3 (Mauve/Widmer 2001 style): forward to neighbor whose predicted
    # position at t + T_lookahead minimizes residual distance to dest,
    # gated by pi_up > p_min.
    if src == dest:
        return [src]
    pred_positions = positions + velocities * T_lookahead
    path = [src]
    cur = src
    visited = {src}
    for _ in range(max_hops):
        if cur == dest:
            return path
        nbrs = np.where(adj[cur])[0]
        nbrs = np.array([n for n in nbrs if n not in visited], dtype=np.int32)
        if len(nbrs) == 0:
            return []
        gate = pi_up[cur, nbrs] > p_min
        if not np.any(gate):
            return []
        dist_d = np.linalg.norm(pred_positions[nbrs] - pred_positions[dest], axis=-1)
        scores = np.where(gate, -dist_d, -np.inf)
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            return []
        nxt = int(nbrs[best])
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    return [] if cur != dest else path
