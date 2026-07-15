"""Genuine-JAX numeric kernels for the GM-PHD recursion (Vo & Ma 2006).

The mixture is carried as fixed-capacity padded arrays so every kernel is a pure
function of arrays and can be ``jit``/``vmap``/``scan``-ed:

    means   : (J, 4)     component means [px, py, vx, vy]
    covs    : (J, 4, 4)  component covariances
    weights : (J,)       component weights (0.0 for padded / invalid slots)
    mask    : (J,)       bool, True for live components

Padded slots carry weight 0 and an identity covariance so they never contribute
to a sum and never make a ``solve``/``slogdet`` blow up.

PSD safety: ``psd_floor`` symmetrizes a matrix and floors its eigenvalues, and
``mvn_pdf_jax`` evaluates the Gaussian density through that floor. A numerically
non-PSD innovation covariance S therefore degrades gracefully (a small but finite
density) instead of raising — the requirement from the audit (models.py:52-54
raised ``ValueError``; the filter's internal path must not).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

# Eigenvalue floor for the PSD-safety projection. Small relative to typical
# measurement covariances; large enough to keep solve/slogdet finite.
_EIG_FLOOR = 1e-9


def psd_floor(P, floor: float = _EIG_FLOOR):
    """Symmetrize ``P`` and floor its eigenvalues at ``floor`` (PSD projection)."""
    Psym = 0.5 * (P + jnp.swapaxes(P, -1, -2))
    w, V = jnp.linalg.eigh(Psym)
    w = jnp.maximum(w, floor)
    return (V * w[..., None, :]) @ jnp.swapaxes(V, -1, -2)


def mvn_pdf_jax(x, mean, cov):
    """PSD-safe multivariate-normal density N(x; mean, cov).

    Floors the covariance eigenvalues before evaluating so a non-PSD ``cov``
    yields a finite density rather than NaN / an error.
    """
    cov = psd_floor(cov)
    d = x - mean
    k = d.shape[-1]
    sign, logdet = jnp.linalg.slogdet(cov)
    sol = jnp.linalg.solve(cov, d)
    return jnp.exp(-0.5 * (d @ sol) - 0.5 * (k * jnp.log(2.0 * jnp.pi) + logdet))


# ----------------------------------------------------------------- predict ----

@partial(jax.jit, static_argnums=())
def predict_kernel(means, covs, weights, F, Q, p_survival):
    """Propagate every component through the CV model (vmapped over components).

    m' = F m ; P' = F P F^T + Q ; w' = p_S w.  (Birth is added in Python so the
    'measurement' / 'intensity' modes share this kernel.)
    """
    new_means = means @ F.T
    new_covs = jax.vmap(lambda P: F @ P @ F.T + Q)(covs)
    new_weights = p_survival * weights
    return new_means, new_covs, new_weights


# ------------------------------------------------------------------ update ----

def _single_update(m, P, z, R, H):
    """Kalman update of one component by one measurement. Returns (m_upd, P_upd, q)."""
    eta = H @ m                         # predicted measurement (2,)
    HP = H @ P                          # (2, 4)
    S = HP @ H.T + R                    # innovation covariance (2, 2)
    S = psd_floor(S)                    # PSD safety
    K = jnp.linalg.solve(S, HP).T       # P H^T S^{-1}  (4, 2)
    m_upd = m + K @ (z - eta)
    P_upd = P - K @ HP                  # (I - K H) P
    P_upd = 0.5 * (P_upd + P_upd.T)     # keep symmetric
    q = mvn_pdf_jax(z, eta, S)
    return m_upd, P_upd, q


@partial(jax.jit, static_argnums=())
def update_kernel(means, covs, weights, mask, zs, Rs, det_mask, H,
                  p_detect, clutter_intensity):
    """Full GM-PHD measurement update on padded arrays.

    Args:
        means/covs/weights/mask : predicted mixture, capacity J
        zs   : (M, 2) measurements, Rs : (M, 2, 2), det_mask : (M,) bool valid
        H, p_detect, clutter_intensity : model params

    Returns padded arrays of capacity J*(M+1):
        out_means (Jo,4), out_covs (Jo,4,4), out_weights (Jo,), out_mask (Jo,)
    The first J slots are the missed-detection terms; the next J*M slots are the
    per-(component, measurement) detection terms.
    """
    J = means.shape[0]
    M = zs.shape[0]
    w_eff = weights * mask  # padded slots contribute 0

    # --- missed-detection terms: (1 - pD) w, mean/cov unchanged ---
    miss_w = (1.0 - p_detect) * w_eff
    miss_means = means
    miss_covs = covs
    miss_mask = mask

    # --- detection terms: vmap over (measurement, component) ---
    # upd[m_idx, j] = update of component j by measurement m
    def per_meas(z, R):
        m_upd, P_upd, q = jax.vmap(lambda mm, PP: _single_update(mm, PP, z, R, H))(
            means, covs)
        w_num = p_detect * w_eff * q        # (J,)
        return m_upd, P_upd, w_num

    det_means, det_covs, det_w_num = jax.vmap(per_meas)(zs, Rs)
    # shapes: det_means (M, J, 4), det_covs (M, J, 4, 4), det_w_num (M, J)

    # normalize each measurement's terms by clutter + sum of that measurement's
    # numerators (over live components only)
    weight_sum = jnp.sum(det_w_num, axis=1)             # (M,)
    denom = clutter_intensity + weight_sum              # (M,)
    det_w = det_w_num / denom[:, None]                  # (M, J)

    # invalid measurements contribute nothing
    det_w = det_w * det_mask[:, None]

    # flatten detection terms (M*J, ...)
    det_means_f = det_means.reshape(M * J, 4)
    det_covs_f = det_covs.reshape(M * J, 4, 4)
    det_w_f = det_w.reshape(M * J)
    det_mask_f = (jnp.ones((M, J), dtype=bool) * det_mask[:, None]).reshape(M * J)
    det_mask_f = det_mask_f & jnp.tile(mask, M)

    out_means = jnp.concatenate([miss_means, det_means_f], axis=0)
    out_covs = jnp.concatenate([miss_covs, det_covs_f], axis=0)
    out_weights = jnp.concatenate([miss_w, det_w_f], axis=0)
    out_mask = jnp.concatenate([miss_mask, det_mask_f], axis=0)
    return out_means, out_covs, out_weights, out_mask


# ------------------------------------------------------------------ prune -----

@jax.jit
def prune_mask(weights, mask, threshold):
    """Mask of components with weight STRICTLY above threshold (and live).

    Vo & Ma 2006 Table II keeps I = {i : w_i > T} — strict inequality."""
    return mask & (weights > threshold)


# ------------------------------------------------------------------ merge -----

def _merge_step(means, covs, weights, alive, U):
    """One greedy-merge iteration (Vo & Ma 2006, Table II).

    Pick the heaviest live component j (the seed). Gather every live component i
    with Mahalanobis^2 (m_i - m_j)^T P_i^{-1} (m_i - m_j) <= U — note P_i is the
    CANDIDATE component's own covariance, not the seed's — moment-match the group,
    and remove the group from the live set.

    Returns (m_bar, P_bar, w_sum, new_alive).
    """
    J = means.shape[0]
    # heaviest live component (masked weight; dead -> -inf so never chosen)
    masked_w = jnp.where(alive, weights, -jnp.inf)
    j = jnp.argmax(masked_w)
    seed_m = means[j]

    diff = means - seed_m                                  # (J, 4), per candidate i
    # d2_i = diff_i^T P_i^{-1} diff_i, using each candidate's OWN covariance P_i
    sol = jnp.linalg.solve(covs, diff[..., None])[..., 0]  # P_i^{-1} diff_i  (J, 4)
    d2 = jnp.sum(diff * sol, axis=-1)                      # (J,)

    in_group = alive & (d2 <= U)
    gw = jnp.where(in_group, weights, 0.0)                 # (J,)
    w_sum = jnp.sum(gw)
    w_safe = jnp.where(w_sum > 0, w_sum, 1.0)

    m_bar = jnp.sum(gw[:, None] * means, axis=0) / w_safe  # (4,)
    spread = means - m_bar                                 # (J, 4)
    outer = covs + spread[:, :, None] * spread[:, None, :]  # (J, 4, 4)
    P_bar = jnp.sum(gw[:, None, None] * outer, axis=0) / w_safe

    new_alive = alive & (~in_group)
    return m_bar, P_bar, w_sum, new_alive


@partial(jax.jit, static_argnums=(4,))
def merge_kernel(means, covs, weights, mask, max_iters, U):
    """Greedy moment-matching merge over a fixed ``max_iters`` budget.

    Runs ``max_iters`` merge steps via ``lax.scan``; each step emits one merged
    component (zero-weight & masked-out when no live components remain). The merge
    gate uses the CANDIDATE component's own covariance (the verified bug fix).
    """
    def body(carry, _):
        alive = carry
        any_alive = jnp.any(alive)
        m_bar, P_bar, w_sum, new_alive = _merge_step(means, covs, weights, alive, U)
        # if nothing was alive this step, emit an inert (masked, zero-weight) slot
        out_m = m_bar
        out_P = jnp.where(any_alive, P_bar, jnp.eye(4))
        out_w = jnp.where(any_alive, w_sum, 0.0)
        out_valid = any_alive
        return new_alive, (out_m, out_P, out_w, out_valid)

    _, (m_out, P_out, w_out, valid_out) = jax.lax.scan(
        body, mask, None, length=max_iters)
    return m_out, P_out, w_out, valid_out


# ------------------------------------------------------------------- cap ------

def cap_indices(weights, mask, J):
    """Indices of the up-to-J heaviest live components, in descending weight."""
    masked_w = jnp.where(mask, weights, -jnp.inf)
    order = jnp.argsort(masked_w)[::-1]
    return order[:J]
