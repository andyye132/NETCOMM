import jax
import jax.numpy as jnp
import jax.scipy.special as jsps
from .nakagami import nakagami_m_t


@jax.jit
def linkup_per_step(sinr_step, gamma_th, m_t):
    threshold_ratio = gamma_th * m_t / (sinr_step + 1e-12)
    p_fail = jsps.gammainc(m_t, threshold_ratio)  # lower regularized
    return 1.0 - p_fail


@jax.jit
def linkup_by_deadline(forecast, gamma_th, r_rng, m_baseline, T_0=1.0):
    sinr = forecast.mean_sinr  # (N, N, L)
    coh = forecast.coh_time   # (N, N, L)
    m_t = nakagami_m_t(coh, m_baseline, T_0)  # (N, N, L)
    p_step = linkup_per_step(sinr, gamma_th, m_t)  # (N, N, L)
    # Approximate range gate using mean SINR floor: SINR << implies dist > r_rng.
    # We multiply by a soft sigmoid on mean SINR vs zero.
    p_step = jnp.clip(p_step, 0.0, 1.0)
    pi_up = jnp.prod(p_step, axis=-1)  # (N, N)
    # Mask self-loops.
    n = pi_up.shape[0]
    pi_up = pi_up * (1.0 - jnp.eye(n, dtype=pi_up.dtype))
    return pi_up


def linkup_by_deadline_per_packet(forecast, delta_p, gamma_th, r_rng,
                                  m_baseline, dt, T_0=1.0):
    # why: per-packet deadlines need only the first ceil(delta_p / dt) horizon
    # slices; we vectorize over a (P,) array of remaining lifetimes by building
    # a (P, L) horizon mask and reducing pi_up over masked slices.
    sinr = forecast.mean_sinr  # (N, N, L)
    coh = forecast.coh_time   # (N, N, L)
    m_t = nakagami_m_t(coh, m_baseline, T_0)  # (N, N, L)
    p_step = linkup_per_step(sinr, gamma_th, m_t)  # (N, N, L)
    p_step = jnp.clip(p_step, 0.0, 1.0)
    L = p_step.shape[-1]
    horizon_idx = jnp.arange(L)  # (L,)
    delta_p = jnp.asarray(delta_p)
    # n_steps[p] = ceil(delta_p[p] / dt), clipped to L.
    n_steps = jnp.clip(jnp.ceil(delta_p / dt).astype(jnp.int32), 1, L)  # (P,)
    # mask[p, l] = 1 if l < n_steps[p] else 0.
    mask = (horizon_idx[None, :] < n_steps[:, None]).astype(p_step.dtype)  # (P, L)
    # Use log-sum to broadcast safely: log_p shape (N, N, L); want prod over masked l.
    log_p = jnp.log(jnp.clip(p_step, 1e-12, 1.0))  # (N, N, L)
    # einsum-style broadcast: out[p, i, j] = sum_l mask[p, l] * log_p[i, j, l]
    log_per_pkt = jnp.einsum("pl,ijl->pij", mask, log_p)  # (P, N, N)
    pi_up = jnp.exp(log_per_pkt)
    n = pi_up.shape[-1]
    pi_up = pi_up * (1.0 - jnp.eye(n, dtype=pi_up.dtype))[None, :, :]
    return pi_up
