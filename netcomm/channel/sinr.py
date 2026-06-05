import jax.numpy as jnp
from .pathloss import path_loss_gain


def sinr_per_edge(positions, P_tx, alpha, beta, N0_B, interference_mode="snr"):
    diff = positions[:, None, :] - positions[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1) + 1e-6
    signal = P_tx * path_loss_gain(dist, alpha, beta)
    if interference_mode == "snr":
        interference = 0.0
    elif interference_mode == "half":
        mask = signal > 0.25 * jnp.max(signal, axis=0, keepdims=True)
        masked = jnp.where(mask, signal, 0.0)
        interference = jnp.sum(masked, axis=0)[None, :] - masked
    else:
        interference = jnp.sum(signal, axis=0)[None, :] - signal
    sinr = signal / (N0_B + interference + 1e-12)
    n = positions.shape[0]
    return sinr * (1.0 - jnp.eye(n, dtype=sinr.dtype))
