from typing import Tuple
import jax.numpy as jnp

from netcomm.types import RegimeBelief, N_REGIMES


def init_belief(N: int, prior: Tuple[float, float, float, float] = (0.6, 0.25, 0.10, 0.05)) -> RegimeBelief:
    p = jnp.asarray(prior, dtype=jnp.float32)
    p = p / jnp.sum(p)
    b = jnp.broadcast_to(p[None, None, :], (N, N, N_REGIMES))
    return RegimeBelief(b=b)


def build_link_transition(dt: float, doppler_severity: jnp.ndarray,
                          blockage_rate: jnp.ndarray, tau: float = 0.5) -> jnp.ndarray:
    # why: a small generative parameterization. base self-loop p_self = exp(-dt/tau)
    # so per-step persistence shrinks smoothly with dt. The off-diagonal mass
    # (1 - p_self) is redistributed by per-cell (doppler_severity, blockage_rate)
    # so that high-Doppler links drift toward "volatile" and high-blockage links
    # drift toward "blocked"; otherwise mass flows back toward "stable".
    p_self = jnp.exp(-dt / max(tau, 1e-6))  # scalar
    N = doppler_severity.shape[0]
    ds = jnp.clip(doppler_severity, 0.0, 1.0)  # (N, N)
    br = jnp.clip(blockage_rate, 0.0, 1.0)  # (N, N)
    # target distribution that the link drifts toward
    # weights over (stable, predictable, volatile, blocked):
    w_stable = (1.0 - ds) * (1.0 - br)
    w_pred = ds * (1.0 - br) * 0.7
    w_vol = ds * (1.0 - br) * 0.3 + ds * br * 0.2
    w_block = br * (1.0 - 0.2 * ds) + br * ds * 0.8
    target = jnp.stack([w_stable, w_pred, w_vol, w_block], axis=-1)  # (N, N, 4)
    target = target / jnp.clip(jnp.sum(target, axis=-1, keepdims=True), 1e-9, None)
    # transition[i, j, k, l] = p_self if k==l else (1-p_self) * target[i, j, l]
    eye = jnp.eye(N_REGIMES, dtype=target.dtype)  # (4, 4)
    off = (1.0 - p_self) * target[..., None, :]  # (N, N, 1, 4) broadcast over from-state
    on = p_self * eye[None, None, :, :]  # (1, 1, 4, 4)
    # off is "to" distribution independent of from-state; replace the diagonal entries
    # so each row sums correctly: row k = p_self * e_k + (1 - p_self) * target.
    off_full = jnp.broadcast_to(off, (N, N, N_REGIMES, N_REGIMES))
    transition = on + off_full * (1.0 - eye[None, None, :, :])
    # normalize for safety
    transition = transition / jnp.clip(jnp.sum(transition, axis=-1, keepdims=True), 1e-9, None)
    return transition


def regime_step(belief: RegimeBelief, obs: jnp.ndarray,
                transition: jnp.ndarray) -> RegimeBelief:
    # why: standard HMM forward step batched over the (N, N) link grid.
    b = belief.b  # (N, N, R)
    # predict: b_pred[i, j, l] = sum_k b[i, j, k] * T[i, j, k, l]
    b_pred = jnp.einsum("ijk,ijkl->ijl", b, transition)
    # update: b_new ∝ obs * b_pred
    unnorm = obs * b_pred
    norm = jnp.clip(jnp.sum(unnorm, axis=-1, keepdims=True), 1e-9, None)
    b_new = unnorm / norm
    return RegimeBelief(b=b_new)
