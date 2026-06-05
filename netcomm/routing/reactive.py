import numpy as np


def gpsr_forward(i: int, dest: int, neighborhood, sinr_row, positions, gamma_th: float):
    if len(neighborhood) == 0:
        return -1
    pos_i = positions[i]
    pos_d = positions[dest]
    in_range = sinr_row[neighborhood] > gamma_th
    dist_i = np.linalg.norm(pos_i - pos_d)
    dist_n = np.linalg.norm(positions[neighborhood] - pos_d, axis=-1)
    geo_progress = np.where(in_range, dist_i - dist_n, -1.0)
    best = int(np.argmax(geo_progress))
    if geo_progress[best] <= 0.0:
        return -1
    return int(neighborhood[best])


def _gabriel_neighbors(u: int, positions, adj, sinr, gamma_th: float):
    # Gabriel Graph planarization (Karp & Kung 2000, sec. 3.2): edge (u,v) kept iff
    # the disk with diameter |uv| contains no other node.
    pos_u = positions[u]
    cand = np.where(adj[u] & (sinr[u] > gamma_th))[0]
    keep = []
    for v in cand:
        if v == u:
            continue
        mid = 0.5 * (pos_u + positions[v])
        r2 = 0.25 * np.sum((pos_u - positions[v]) ** 2)
        d2 = np.sum((positions[cand] - mid) ** 2, axis=-1)
        if not np.any((d2 < r2 - 1e-12) & (cand != v) & (cand != u)):
            keep.append(int(v))
    return keep


def _rhr_next(cur: int, prev: int, positions, planar_nbrs):
    # Right-hand rule: at `cur` arriving from `prev`, pick the next neighbor
    # encountered by sweeping clockwise from the (cur->prev) bearing.
    pos_c = positions[cur]
    ref = np.arctan2(positions[prev][1] - pos_c[1], positions[prev][0] - pos_c[0])
    best_v, best_delta = -1, np.inf
    for v in planar_nbrs:
        if v == cur:
            continue
        ang = np.arctan2(positions[v][1] - pos_c[1], positions[v][0] - pos_c[0])
        delta = (ref - ang) % (2 * np.pi)
        if delta < 1e-12:
            delta = 2 * np.pi  # avoid picking prev itself
        if delta < best_delta:
            best_delta, best_v = delta, int(v)
    return best_v


def gpsr_route(src: int, dest: int, positions, adj, sinr, gamma_th: float,
               max_hops: int = 16):
    pos_d = positions[dest]
    path = [src]
    cur = src
    mode = "greedy"
    perim_entry_dist = None
    prev = -1
    for _ in range(max_hops):
        if cur == dest:
            return path
        if mode == "greedy":
            nbrs = np.where(adj[cur])[0]
            nxt = gpsr_forward(cur, dest, nbrs, sinr[cur], positions, gamma_th)
            if nxt >= 0:
                prev = cur
                path.append(nxt)
                cur = nxt
                continue
            mode = "perimeter"
            perim_entry_dist = np.linalg.norm(positions[cur] - pos_d)
            # Bootstrap face traversal by treating the destination direction as the
            # incoming edge so the first hop is the CCW-most neighbor about (cur->dest).
            prev = -1
        planar = _gabriel_neighbors(cur, positions, adj, sinr, gamma_th)
        if not planar:
            return []
        if prev < 0:
            # Synthesize a virtual "previous" along the cur->dest ray.
            virt = pos_d
            ref = np.arctan2(virt[1] - positions[cur][1], virt[0] - positions[cur][0])
            best_v, best_delta = -1, np.inf
            for v in planar:
                ang = np.arctan2(positions[v][1] - positions[cur][1],
                                 positions[v][0] - positions[cur][0])
                delta = (ang - ref) % (2 * np.pi)
                if delta < best_delta:
                    best_delta, best_v = delta, v
            nxt = best_v
        else:
            nxt = _rhr_next(cur, prev, positions, planar)
        if nxt < 0:
            return []
        prev = cur
        path.append(nxt)
        cur = nxt
        if np.linalg.norm(positions[cur] - pos_d) < perim_entry_dist:
            mode = "greedy"
            prev = -1
    return []


def glsr_forward(i: int, dest: int, neighborhood, sinr_row, pi_row,
                 positions, gamma_th: float):
    if len(neighborhood) == 0:
        return -1
    pos_d = positions[dest]
    dist_i = np.linalg.norm(positions[i] - pos_d)
    dist_n = np.linalg.norm(positions[neighborhood] - pos_d, axis=-1)
    progress = np.clip(dist_i - dist_n, 0.0, None)
    in_range = sinr_row[neighborhood] > gamma_th
    # GLSR (Ducourthial 2007): forward to neighbor maximizing progress * stability.
    # pi_up (forecast link-up probability) is our link-stability proxy.
    score = np.where(in_range, progress * pi_row[neighborhood], -1.0)
    best = int(np.argmax(score))
    if score[best] <= 0.0:
        return -1
    return int(neighborhood[best])


def glsr_route(src: int, dest: int, positions, adj, sinr, pi_up,
               gamma_th: float, max_hops: int = 16):
    path = [src]
    cur = src
    visited = {src}
    for _ in range(max_hops):
        if cur == dest:
            return path
        nbrs = np.where(adj[cur])[0]
        nbrs = np.array([n for n in nbrs if n not in visited], dtype=np.int32)
        nxt = glsr_forward(cur, dest, nbrs, sinr[cur], pi_up[cur],
                           positions, gamma_th)
        if nxt < 0:
            return []
        path.append(int(nxt))
        visited.add(int(nxt))
        cur = int(nxt)
    return [] if cur != dest else path


def aodv_forward(src: int, dest: int, positions, adj, sinr, gamma_th: float,
                 max_hops: int = 16):
    n = adj.shape[0]
    if src == dest:
        return [src]
    parent = {src: -1}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        if cur == dest:
            path = [cur]
            while parent[path[-1]] != -1:
                path.append(parent[path[-1]])
            return list(reversed(path))
        for j in range(n):
            if not adj[cur, j] or j in parent:
                continue
            if sinr[cur, j] <= gamma_th:
                continue
            parent[j] = cur
            queue.append(j)
    return []


def dsr_forward(src: int, dest: int, positions, adj, sinr, gamma_th: float,
                max_hops: int = 16, cache=None, step: int = 0, ttl: int = 8):
    # DSR's defining feature vs AODV: cached source routes are reused without
    # rediscovery until they expire or the underlying link drops. We deliberately
    # skip the SINR re-gate within ttl so stale cache entries can mis-route —
    # that is the published DSR weakness in mobile scenarios (Maltz et al. 1999).
    if cache is not None:
        hit = cache.get((src, dest))
        if hit is not None:
            path, learned = hit
            edges_exist = all(adj[path[k], path[k + 1]]
                              for k in range(len(path) - 1))
            if edges_exist and step - learned < ttl:
                return path
    path = aodv_forward(src, dest, positions, adj, sinr, gamma_th, max_hops)
    if cache is not None and path:
        cache[(src, dest)] = (path, step)
    return path
