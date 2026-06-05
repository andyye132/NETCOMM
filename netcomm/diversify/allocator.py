from typing import List

import numpy as np
import jax.numpy as jnp


def greedy_allocate(F: int, paths: List, costs: np.ndarray) -> np.ndarray:
    # why: top-k assignment by -log(LCB). costs has shape (P,).
    P = len(paths)
    out = np.zeros((F, P), dtype=np.int32)
    if P == 0 or F == 0:
        return out
    order = np.argsort(costs)  # ascending cost
    for f in range(F):
        p = int(order[f % P])
        out[f, p] = 1
    return out


def sinkhorn_allocate(F: int, paths: List, costs: np.ndarray,
                      n_iter: int = 50, eps: float = 0.1) -> np.ndarray:
    # why: entropic optimal transport between F fragments (uniform mass 1/F)
    # and P paths (uniform mass 1/P). Returns a soft (F, P) coupling.
    P = len(paths)
    if P == 0 or F == 0:
        return np.zeros((F, P), dtype=np.float32)
    c = jnp.asarray(costs, dtype=jnp.float32)  # (P,)
    K = jnp.exp(-jnp.broadcast_to(c[None, :], (F, P)) / max(eps, 1e-3))
    a = jnp.ones((F,), dtype=jnp.float32) / F
    b = jnp.ones((P,), dtype=jnp.float32) / P
    u = jnp.ones((F,), dtype=jnp.float32)
    v = jnp.ones((P,), dtype=jnp.float32)
    for _ in range(int(n_iter)):
        u = a / jnp.clip(K @ v, 1e-9, None)
        v = b / jnp.clip(K.T @ u, 1e-9, None)
    coupling = u[:, None] * K * v[None, :]
    return np.asarray(coupling)
