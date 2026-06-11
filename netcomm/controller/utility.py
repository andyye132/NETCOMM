import heapq
from typing import List, Tuple

import numpy as np

from netcomm.types import Packet, NetCommConfig
from netcomm.diversify.paths import k_disjoint_paths
from netcomm.diversify.erasure import k_of_n_decode_prob


def _lagrangian(deliver_prob: float, costs, cfg: NetCommConfig) -> float:
    B, E, C, Q = costs
    return (deliver_prob
            - cfg.lambda_B * B
            - cfg.lambda_E * E
            - cfg.lambda_C * C
            - cfg.lambda_Q * Q)


def _greedy_path(src: int, dst: int, pi_up_np: np.ndarray,
                 positions: np.ndarray, adj: np.ndarray,
                 max_hops: int = 16) -> List[int]:
    cur = int(src)
    path = [cur]
    visited = {cur}
    for _ in range(max_hops):
        if cur == dst:
            return path
        nbrs = np.where(adj[cur])[0]
        nbrs = np.array([n for n in nbrs if n not in visited], dtype=np.int32)
        if len(nbrs) == 0:
            return []
        dist_cur = np.linalg.norm(positions[cur] - positions[dst])
        dist_n = np.linalg.norm(positions[nbrs] - positions[dst], axis=-1)
        progress = np.clip(dist_cur - dist_n, 0.0, None)
        score = pi_up_np[cur, nbrs] * progress
        best = int(np.argmax(score))
        if score[best] <= 0.0:
            return []
        cur = int(nbrs[best])
        path.append(cur)
        visited.add(cur)
    return path if path[-1] == dst else []


def _path_survival(path: List[int], values: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    p = 1.0
    for u, v in zip(path[:-1], path[1:]):
        p *= float(values[u, v])
    return p


def _dijkstra_log_cost(src: int, dst: int, adj: np.ndarray,
                       values: np.ndarray, max_hops: int = 16) -> List[int]:
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
        if u == dst:
            break
        if depth[u] >= max_hops:
            continue
        for v in range(n):
            if not adj[u, v] or v == u:
                continue
            p = float(values[u, v])
            if p <= 1e-9:
                continue
            nc = c - np.log(max(p, 1e-9))
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


def _belief_weights(belief_local) -> Tuple[float, float, float, float]:
    # why: collapse (N, 4) per-edge belief to a 4-vector by averaging over
    # outgoing edges of the source. Matches netcomm.tex eq. for b_t.
    b = np.asarray(belief_local, dtype=np.float64)
    if b.ndim == 2:
        b = np.mean(b, axis=0)
    b = b / max(float(np.sum(b)), 1e-9)
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def U_react(packet: Packet, belief_local, pi_up, sinr, positions, adj,
            cfg: NetCommConfig, costs) -> Tuple[float, List[int]]:
    pi_up_np = np.asarray(pi_up)
    pos_np = np.asarray(positions)
    adj_np = np.asarray(adj)
    path = _greedy_path(int(packet.src), int(packet.dst), pi_up_np, pos_np, adj_np)
    if not path or path[-1] != int(packet.dst):
        return _lagrangian(0.0, costs, cfg), path
    surv = _path_survival(path, pi_up_np)
    n_hops = len(path) - 1
    delta_p = float(packet.deadline - packet.t_gen)
    meets = 1.0 if n_hops * cfg.dt < max(delta_p, 1e-6) else 0.0
    w_st, w_pr, _, _ = _belief_weights(belief_local)
    # snapshot-SINR is trustworthy in stable regime, partially in predictable.
    # Floor at 0.4 so a mismatched belief doesn't zero out a viable action.
    w = 0.4 + 0.6 * min(1.0, w_st + 0.5 * w_pr)
    return _lagrangian(surv * meets * w, costs, cfg), path


def U_predict(packet: Packet, belief_local, lcb, adj, cfg: NetCommConfig,
              costs) -> Tuple[float, List[int]]:
    lcb_np = np.asarray(lcb)
    adj_np = np.asarray(adj)
    path = _dijkstra_log_cost(int(packet.src), int(packet.dst), adj_np, lcb_np)
    if not path or path[-1] != int(packet.dst):
        return _lagrangian(0.0, costs, cfg), path
    surv = _path_survival(path, lcb_np)
    n_hops = len(path) - 1
    delta_p = float(packet.deadline - packet.t_gen)
    meets = 1.0 if n_hops * cfg.dt < max(delta_p, 1e-6) else 0.0
    w_st, w_pr, w_vo, _ = _belief_weights(belief_local)
    w = 0.4 + 0.6 * min(1.0, w_pr + 0.5 * w_st + 0.25 * w_vo)
    return _lagrangian(surv * meets * w, costs, cfg), path


def U_diversify(packet: Packet, belief_local, lcb, adj, cfg: NetCommConfig,
                costs, k_paths: int) -> Tuple[float, List[List[int]]]:
    lcb_np = np.asarray(lcb)
    adj_np = np.asarray(adj)
    edge_cost = -np.log(np.clip(lcb_np, 1e-9, 1.0))
    paths = k_disjoint_paths(adj_np, int(packet.src), int(packet.dst),
                             int(k_paths), edge_cost)
    if not paths:
        return _lagrangian(0.0, costs, cfg), []
    survs = np.array([_path_survival(p, lcb_np) for p in paths], dtype=np.float64)
    n_paths = len(paths)
    k_dec = int(min(cfg.k_decode, n_paths))
    n_frag = int(max(cfg.n_fragments, n_paths))
    s_rep = np.array([survs[i % n_paths] for i in range(n_frag)], dtype=np.float64)
    p_decode = k_of_n_decode_prob(s_rep, k_dec, n_frag)
    n_hops_max = max((len(p) - 1) for p in paths)
    delta_p = float(packet.deadline - packet.t_gen)
    meets = 1.0 if n_hops_max * cfg.dt < max(delta_p, 1e-6) else 0.0
    _, w_pr, w_vo, _ = _belief_weights(belief_local)
    w = 0.4 + 0.6 * min(1.0, w_vo + 0.5 * w_pr)
    return _lagrangian(p_decode * meets * w, costs, cfg), paths


def U_drop(packet: Packet, belief_local=None) -> float:
    # why: drop wins only when forwarding utilities all drop below ~0.05.
    # A flat w_block coefficient (1.0) made oracle_regime drop everything in
    # NLoS scenarios; a small constant keeps drop in play without dominating.
    if belief_local is None:
        return 0.0
    _, _, _, w_bl = _belief_weights(belief_local)
    return 0.05 * float(w_bl)


def value_of_prediction(U_p: float, U_r: float) -> float:
    return float(U_p - U_r)


def value_of_diversification(U_d: float, U_r: float, U_p: float) -> float:
    return float(U_d - max(U_r, U_p))
