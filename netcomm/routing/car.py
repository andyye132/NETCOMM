import numpy as np


def car_route(src, dest, positions, adj, pi_up, ctx, p_min, max_hops=16):
    # CAR (Musolesi/Mascolo 2009 style): forward to neighbor maximizing
    # ctx_j * pi_up(i, j) subject to geographic progress toward dest.
    if src == dest:
        return [src]
    pos_d = positions[dest]
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
        dist_cur = np.linalg.norm(positions[cur] - pos_d)
        dist_n = np.linalg.norm(positions[nbrs] - pos_d, axis=-1)
        progress = dist_cur - dist_n
        pi_n = pi_up[cur, nbrs]
        gate = (pi_n > p_min) & (progress > 0.0)
        if not np.any(gate):
            return []
        score = np.where(gate, ctx[nbrs] * pi_n, -1.0)
        best = int(np.argmax(score))
        if score[best] <= 0.0:
            return []
        nxt = int(nbrs[best])
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    return [] if cur != dest else path
