"""Isotropic Voronoi coverage control — the pure-JAX Lloyd recursion (Cortes 2004).

Self-contained (imports nothing from netcomm). Implements the locational
optimization of

    H(P) = sum_i  integral_{V_i} ||q - p_i||^2 phi(q) dq                      (cost)

over sensor positions P = (p_1, ..., p_n) in a convex region Q with importance
density phi, via the two coupled optimalities of Cortes, Martinez, Karatas &
Bullo, "Coverage Control for Mobile Sensing Networks", IEEE T-RO 2004:

  * optimal partition  = the Voronoi partition  V_i = {q : ||q-p_i|| <= ||q-p_j||}
  * optimal location   = the phi-weighted centroid  C_{V_i} = (int q phi) / (int phi)

The gradient is  dH/dp_i = 2 M_{V_i} (p_i - C_{V_i})  (the Voronoi-boundary
Leibniz terms cancel because adjacent cells share an equidistant face), so the
control law  u_i = -k (p_i - C_{V_i})  is gradient descent and H is a Lyapunov
function (H decreases to a centroidal Voronoi configuration).

The integrals are evaluated by GRID QUADRATURE: a regular grid of Q is assigned
to the nearest sensor (an argmin = the Voronoi partition), and per-cell mass,
centroid, and cost are segment reductions. Grid resolution is the quadrature
knob; everything converges to the exact integrals as it grows.

Anisotropy (Gusrialdi et al. 2008) is the F != I generalization: pre-map the
grid qbar = F q, run THIS isotropic core in qbar-space, post-map C = F^{-1} Cbar.
This module is exactly the F = I case, so that drops in later with no rewrite.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def make_grid(area_xy, nx: int, ny: int):
    """Regular cell-center quadrature grid over the rectangle Q = area_xy.

    Returns (grid_pts (G, 2) with G = nx*ny, cell_area scalar). Uses the same
    cell-center convention as netcomm.tracking.coverage.CoverageField so the phi
    grid aligns with the existing heatmap on screen.
    """
    xmn, xmx, ymn, ymx = (float(a) for a in area_xy)
    xs = xmn + (jnp.arange(nx) + 0.5) / nx * (xmx - xmn)
    ys = ymn + (jnp.arange(ny) + 0.5) / ny * (ymx - ymn)
    gx, gy = jnp.meshgrid(xs, ys)                                  # (ny, nx)
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=1)             # (G, 2), row 0 = ymin
    cell_area = ((xmx - xmn) / nx) * ((ymx - ymn) / ny)
    return pts, float(cell_area)


def build_phi_grid(grid_pts, centers, weights, sigma: float, floor: float):
    """Rasterize the importance density phi onto the grid.

    phi(q) = floor + sum_m w_m * exp(-||q - c_m||^2 / (2 sigma^2)).

    centers (M, 2) are the target-estimate means, weights (M,) their PHD weights.
    With no centers, phi reduces to the uniform floor (pure area coverage / cold
    start). phi is left UN-normalized: only the centroid (a ratio) and the
    descent direction matter, both scale-invariant in phi.
    """
    G = grid_pts.shape[0]
    centers = jnp.asarray(centers, dtype=float).reshape(-1, 2)
    if centers.shape[0] == 0:
        return jnp.full((G,), float(floor))
    weights = jnp.asarray(weights, dtype=float).reshape(-1)
    d2 = ((grid_pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1)   # (G, M)
    bumps = weights[None, :] * jnp.exp(-d2 / (2.0 * sigma * sigma))    # (G, M)
    return floor + bumps.sum(axis=1)                                  # (G,)


def voronoi_assign(positions, grid_pts):
    """Nearest-sensor label per grid cell = the Voronoi partition of Q.

    jnp.argmin breaks ties to the lowest index; ties are a measure-zero set so
    this does not affect masses/centroids/cost.
    """
    d2 = ((grid_pts[:, None, :] - positions[None, :, :]) ** 2).sum(-1)   # (G, N)
    return jnp.argmin(d2, axis=1)                                        # (G,)


@partial(jax.jit, static_argnums=(4,))
def partition_centroids_cost(positions, phi, grid_pts, cell_area, n: int):
    """One Voronoi partition + per-cell mass, centroid, and the coverage cost.

    Returns (labels (G,), mass (n,), centroids (n, 2), cost scalar):
        M_i = sum_{cell in V_i} phi * cell_area
        C_i = (sum phi * cell_area * q) / M_i          (current position if V_i empty)
        H   = sum_cells phi * cell_area * ||q - p_{label}||^2

    Empty cells (M_i = 0) keep the sensor in place, which makes its control term
    u_i = -k(p_i - C_i) = 0 — consistent with M_i zeroing that gradient term.
    """
    labels = voronoi_assign(positions, grid_pts)                  # (G,)
    wphi = phi * cell_area                                        # (G,) quadrature weights
    onehot = (labels[:, None] == jnp.arange(n)[None, :]).astype(wphi.dtype)   # (G, n)
    M = (wphi[:, None] * onehot).sum(axis=0)                      # (n,)
    S = onehot.T @ (wphi[:, None] * grid_pts)                     # (n, 2)  = sum phi*area*q
    empty = M <= 0.0
    M_safe = jnp.where(empty, 1.0, M)
    C = S / M_safe[:, None]                                       # (n, 2)
    C = jnp.where(empty[:, None], positions, C)                   # empty cell: stay put
    d2 = ((grid_pts - positions[labels]) ** 2).sum(-1)           # (G,)
    cost = jnp.sum(wphi * d2)
    return labels, M, C, cost


def coverage_cost(positions, phi, grid_pts, cell_area):
    """The Cortes locational cost H(P) (re-partitions internally)."""
    labels = voronoi_assign(positions, grid_pts)
    d2 = ((grid_pts - positions[labels]) ** 2).sum(-1)
    return jnp.sum(phi * cell_area * d2)


def lloyd_step(positions, phi, grid_pts, cell_area, n: int):
    """Discrete Lloyd update: jump each (non-empty) sensor to its cell centroid.

    Returns (new_positions (n, 2), cost_before scalar, mass (n,), centroids (n, 2)).
    Monotone: cost(new) <= cost(positions).
    """
    labels, M, C, cost = partition_centroids_cost(positions, phi, grid_pts, cell_area, n)
    return C, cost, M, C


def capped_descent(positions, centroids, gain: float, dt: float, v_max: float):
    """The Cortes eq. 10 update given precomputed centroids:
    p <- p + dt * clip(-k(p - C), v_max), the speed cap scaling the whole
    displacement so the descent DIRECTION (toward the centroid) is preserved.

    THE single home of this formula — control_step (whose Lyapunov monotone
    descent is pinned by unit tests) and CoverageController.step (production)
    both call it, so the tested formula IS the shipped formula."""
    u = gain * (centroids - positions)                           # = -k (p - C)
    speed = jnp.linalg.norm(u, axis=1, keepdims=True)
    scale = jnp.minimum(1.0, v_max / jnp.maximum(speed, 1e-9))
    return positions + dt * (u * scale)


def control_step(positions, phi, grid_pts, cell_area, n: int,
                 gain: float, dt: float, v_max: float):
    """Continuous Lloyd descent (Cortes eq. 10): p <- p + dt * clip(-k(p-C), v_max).

    Returns (new_positions (n, 2), cost_before, mass (n,), centroids (n, 2)).
    """
    labels, M, C, cost = partition_centroids_cost(positions, phi, grid_pts, cell_area, n)
    return capped_descent(positions, C, gain, dt, v_max), cost, M, C
