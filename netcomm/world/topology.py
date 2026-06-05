import numpy as np


def build_graph(positions, r_rng: float, valid_mask):
    pos = np.asarray(positions)
    valid = np.asarray(valid_mask, dtype=bool)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    in_range = (dist > 0.0) & (dist <= r_rng)
    pair_valid = valid[:, None] & valid[None, :]
    adj = in_range & pair_valid
    np.fill_diagonal(adj, False)
    return adj


def neighbors_of(i: int, adj):
    return np.where(adj[i])[0]
