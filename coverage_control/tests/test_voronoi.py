"""Correctness tests for isotropic Voronoi coverage control (Cortes et al. 2004).

Anchored to closed-form results and to the convergence/cost behavior reported in
the papers. The math is exercised in float64 (x64) so the closed-form and
finite-difference-gradient checks are tight; the live sim runs the same code in
default float32 for speed.

Reproduction note (paper-faithful validation): the source papers run uniform
phi = 1 in their simulations (Gusrialdi et al. 2008 uses phi(q)=1 in every
figure), so the convergence/cost tests here use phi = 1. The live tracking sim
instead drives phi from GM-PHD estimates — the controller is phi-agnostic, so it
is the same code path with a different phi.
"""
import jax
jax.config.update("jax_enable_x64", True)          # tight numerics for the math checks

import numpy as np
import jax.numpy as jnp
import pytest

from coverage_control import (
    make_grid, build_phi_grid, partition_centroids_cost, coverage_cost, control_step,
)


# --------------------------------------------------------------------------- helpers
def uniform_phi(pts):
    return jnp.ones(pts.shape[0], dtype=pts.dtype)


def run_lloyd(P0, phi, pts, ca, n, steps):
    """Discrete Lloyd: jump to centroid each step. Returns (final P, cost history)."""
    P = jnp.asarray(P0, dtype=float)
    costs = []
    for _ in range(steps):
        _labels, _M, C, cost = partition_centroids_cost(P, phi, pts, ca, n)
        costs.append(float(cost))
        P = C
    return np.asarray(P), costs


def residual(P, phi, pts, ca, n):
    _labels, _M, C, _cost = partition_centroids_cost(jnp.asarray(P), phi, pts, ca, n)
    return float(jnp.max(jnp.linalg.norm(jnp.asarray(P) - C, axis=1)))


def match_sort(P):
    return np.array(sorted([tuple(np.round(p, 4)) for p in np.asarray(P)]))


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("W,H", [(6.0, 4.0), (10.0, 10.0)])
def test_single_sensor_uniform_rectangle_closed_form(W, H):
    """1 sensor, uniform phi: centroid = rectangle center, mass = W*H,
    minimal cost (sensor at centroid) = W*H*(W^2+H^2)/12 (polar 2nd moment)."""
    pts, ca = make_grid((0.0, W, 0.0, H), 240, 240)
    phi = uniform_phi(pts)
    P = jnp.array([[W / 2, H / 2]])
    _labels, M, C, cost = partition_centroids_cost(P, phi, pts, ca, 1)
    assert np.allclose(np.asarray(C[0]), [W / 2, H / 2], atol=1e-6)
    assert abs(float(M[0]) - W * H) / (W * H) < 1e-6
    expected = W * H * (W ** 2 + H ** 2) / 12.0
    assert abs(float(cost) - expected) / expected < 1e-3


def test_nonuniform_phi_linear_ramp_closed_form():
    """phi(q) = q_x (linear ramp) on [0,W]x[0,H], single sensor: proves the centroid
    and cost are genuinely phi-WEIGHTED (a phi-dropping bug passes every uniform test).

    Closed forms:  M = int x dA = W^2 H / 2;  centroid = (2W/3, H/2);
    cost at the centroid = int x*||q-C||^2 dA = H W^4 / 36 + W^2 H^3 / 24.
    """
    W, H = 6.0, 4.0
    pts, ca = make_grid((0.0, W, 0.0, H), 300, 200)
    phi = pts[:, 0]                                    # phi(q) = x   (non-uniform)
    C_true = np.array([2 * W / 3, H / 2])             # (4.0, 2.0)
    P = jnp.array([C_true])                            # sensor at the phi-centroid
    _labels, M, C, cost = partition_centroids_cost(P, phi, pts, ca, 1)
    assert abs(float(M[0]) - W ** 2 * H / 2) / (W ** 2 * H / 2) < 1e-3   # mass = 72
    assert np.allclose(np.asarray(C[0]), C_true, atol=2e-3)              # centroid = (4, 2)
    cost_true = H * W ** 4 / 36.0 + W ** 2 * H ** 3 / 24.0               # = 240.0
    assert abs(float(cost) - cost_true) / cost_true < 3e-3


def test_total_mass_equals_area():
    pts, ca = make_grid((0.0, 10.0, 0.0, 10.0), 200, 200)
    phi = uniform_phi(pts)
    P = jnp.array([[2.0, 2.0], [8.0, 3.0], [5.0, 8.0]])
    _labels, M, _C, _cost = partition_centroids_cost(P, phi, pts, ca, 3)
    assert abs(float(jnp.sum(M)) - 100.0) < 1e-6


def test_empty_cell_keeps_sensor_in_place():
    """Two coincident sensors: argmin gives all cells to the lower index; the other
    cell is empty (M=0) and its centroid must equal its own position (no motion)."""
    pts, ca = make_grid((0.0, 10.0, 0.0, 10.0), 120, 120)
    phi = uniform_phi(pts)
    P = jnp.array([[5.0, 5.0], [5.0, 5.0]])          # identical positions
    _labels, M, C, _cost = partition_centroids_cost(P, phi, pts, ca, 2)
    assert float(M[1]) == 0.0
    assert np.allclose(np.asarray(C[1]), [5.0, 5.0])  # empty -> stay put


def test_lyapunov_monotone_descent_lloyd():
    """Cortes cost H is non-increasing every Lloyd step (Lyapunov / LaSalle)."""
    pts, ca = make_grid((0.0, 10.0, 0.0, 10.0), 160, 160)
    phi = uniform_phi(pts)
    rng = np.random.default_rng(0)
    P0 = rng.uniform([0, 0], [10, 10], size=(6, 2))
    _P, costs = run_lloyd(P0, phi, pts, ca, 6, 40)
    c = np.array(costs)
    assert np.all(np.diff(c) <= 1e-6 * c[0])          # monotone non-increasing
    assert c[-1] < c[0]                                # genuine descent


def test_lyapunov_monotone_descent_control_law():
    """The continuous control law u=-k(p-C), speed-capped Euler, also descends H."""
    pts, ca = make_grid((0.0, 10.0, 0.0, 10.0), 160, 160)
    phi = uniform_phi(pts)
    rng = np.random.default_rng(3)
    P = jnp.asarray(rng.uniform([0, 0], [10, 10], size=(5, 2)))
    costs = []
    for _ in range(80):
        new, cost, _M, _C = control_step(P, phi, pts, ca, 5, gain=1.0, dt=0.05, v_max=10.0)
        costs.append(float(cost))
        P = new
    c = np.array(costs)
    assert np.all(np.diff(c) <= 1e-6 * c[0])
    assert c[-1] < 0.9 * c[0]


def test_fixed_point_centroidal_config_no_motion():
    """Starting AT the quadrant-centroid CVT, sensors are already at their centroids."""
    W, H = 8.0, 6.0
    pts, ca = make_grid((0.0, W, 0.0, H), 200, 200)
    phi = uniform_phi(pts)
    P = jnp.array([[W / 4, H / 4], [3 * W / 4, H / 4],
                   [W / 4, 3 * H / 4], [3 * W / 4, 3 * H / 4]])
    assert residual(P, phi, pts, ca, 4) < 0.05         # < ~half a grid cell


def test_convergence_from_random_init_reaches_cvt():
    """From random init, Lloyd drives the max centroid residual to ~0 (a CVT)."""
    pts, ca = make_grid((0.0, 10.0, 0.0, 10.0), 160, 160)
    phi = uniform_phi(pts)
    rng = np.random.default_rng(1)
    P0 = rng.uniform([0, 0], [10, 10], size=(8, 2))
    e0 = residual(P0, phi, pts, ca, 8)
    Pf, _costs = run_lloyd(P0, phi, pts, ca, 8, 200)
    ef = residual(Pf, phi, pts, ca, 8)
    assert ef < 0.05
    assert ef < 0.1 * e0


def test_four_sensors_converge_to_quadrant_centroids():
    """n=4 on a uniform rectangle converges to the 2x2 quadrant-centroid CVT,
    each cell of mass W*H/4."""
    W, H = 8.0, 6.0
    pts, ca = make_grid((0.0, W, 0.0, H), 200, 200)
    phi = uniform_phi(pts)
    quads = np.array([[W / 4, H / 4], [3 * W / 4, H / 4],
                      [W / 4, 3 * H / 4], [3 * W / 4, 3 * H / 4]])
    rng = np.random.default_rng(2)
    P0 = quads + rng.uniform(-0.6, 0.6, size=(4, 2))   # start in the quadrant basin
    Pf, _costs = run_lloyd(P0, phi, pts, ca, 4, 300)
    assert np.allclose(match_sort(Pf), match_sort(quads), atol=0.05)
    _labels, M, _C, _cost = partition_centroids_cost(jnp.asarray(Pf), phi, pts, ca, 4)
    assert np.allclose(np.asarray(M), W * H / 4, rtol=2e-2)


def test_gradient_equals_2M_p_minus_C():
    """Analytic partition-aware gradient dH/dp_i = 2 M_i (p_i - C_i) matches the
    central finite difference of H (Voronoi boundary Leibniz terms cancel)."""
    pts, ca = make_grid((0.0, 10.0, 0.0, 10.0), 300, 300)
    phi = uniform_phi(pts)
    P = jnp.array([[3.0, 3.0], [7.0, 4.0], [5.0, 8.0]])
    _labels, M, C, _cost = partition_centroids_cost(P, phi, pts, ca, 3)
    g_analytic = 2.0 * np.asarray(M)[:, None] * (np.asarray(P) - np.asarray(C))

    h = 1e-3
    g_fd = np.zeros((3, 2))
    for i in range(3):
        for ax in range(2):
            Pp = P.at[i, ax].add(h)
            Pm = P.at[i, ax].add(-h)
            g_fd[i, ax] = float((coverage_cost(Pp, phi, pts, ca)
                                 - coverage_cost(Pm, phi, pts, ca)) / (2 * h))
    for i in range(3):
        denom = np.linalg.norm(g_analytic[i]) + 1e-9
        assert np.linalg.norm(g_analytic[i] - g_fd[i]) / denom < 2e-2


def test_grid_quadrature_matches_shapely_exact_voronoi():
    """For uniform phi, grid masses/centroids match exact bounded-Voronoi polygon
    area/centroid (independent shapely half-plane construction)."""
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon

    area = (0.0, 10.0, 0.0, 10.0)
    Q = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    gens = np.array([[2.0, 2.5], [7.5, 2.0], [5.0, 5.5], [2.5, 8.0], [8.0, 8.5]])
    n = len(gens)

    def halfplane(pi, pj):
        # { x : closer to pi than pj } = { x : dot(n,x) <= c },  n = pj-pi
        nvec = pj - pi
        c = (pj @ pj - pi @ pi) / 2.0
        norm = np.hypot(*nvec)
        nhat = nvec / norm
        x0 = (c / norm) * nhat
        d = np.array([-nhat[1], nhat[0]])
        L = 1e4
        return Polygon([x0 + L * d, x0 - L * d, x0 - L * d - L * nhat, x0 + L * d - L * nhat])

    poly_mass, poly_cent = [], []
    for i in range(n):
        cell = Q
        for j in range(n):
            if j != i:
                cell = cell.intersection(halfplane(gens[i], gens[j]))
        poly_mass.append(cell.area)
        poly_cent.append([cell.centroid.x, cell.centroid.y])
    poly_mass = np.array(poly_mass)
    poly_cent = np.array(poly_cent)

    pts, ca = make_grid(area, 300, 300)
    phi = uniform_phi(pts)
    _labels, M, C, _cost = partition_centroids_cost(jnp.asarray(gens), phi, pts, ca, n)
    M, C = np.asarray(M), np.asarray(C)

    assert abs(M.sum() - 100.0) < 1e-6
    assert np.allclose(M, poly_mass, rtol=5e-3, atol=5e-3)
    for i in range(n):
        assert np.linalg.norm(C[i] - poly_cent[i]) < 0.05


def test_control_law_is_local_to_delaunay_neighbors():
    """Cortes' isotropic law is distributed: a sensor's centroid depends only on its
    Delaunay neighbors. Moving a far, non-adjacent sensor must not change it."""
    pts, ca = make_grid((0.0, 100.0, 0.0, 100.0), 200, 200)
    phi = uniform_phi(pts)
    base = np.array([[10.0, 50.0], [30.0, 50.0], [50.0, 50.0],
                     [70.0, 50.0], [90.0, 50.0]])      # a row; s0 and s4 are NOT neighbors
    _l, _M, C0, _c = partition_centroids_cost(jnp.asarray(base), phi, pts, ca, 5)
    moved = base.copy()
    moved[0] = [10.0, 20.0]                            # move s0 within its own region
    _l, _M, C1, _c = partition_centroids_cost(jnp.asarray(moved), phi, pts, ca, 5)
    assert np.linalg.norm(np.asarray(C0[4]) - np.asarray(C1[4])) < 1e-6   # s4 unaffected
    assert np.linalg.norm(np.asarray(C0[0]) - np.asarray(C1[0])) > 1.0    # s0 did change


def test_no_collision_during_convergence():
    """Collision-free guarantee (Gusrialdi Prop. 4.2): sensors never coincide while
    converging, since distinct generators have distinct (non-overlapping) cells."""
    pts, ca = make_grid((0.0, 10.0, 0.0, 10.0), 120, 120)
    phi = uniform_phi(pts)
    rng = np.random.default_rng(5)
    P = jnp.asarray(rng.uniform([0, 0], [10, 10], size=(6, 2)))
    for _ in range(60):
        _l, _M, C, _c = partition_centroids_cost(P, phi, pts, ca, 6)
        P = C
        Pn = np.asarray(P)
        d = np.linalg.norm(Pn[:, None, :] - Pn[None, :, :], axis=-1)
        d[np.diag_indices(6)] = np.inf
        assert d.min() > 1e-3                          # no two sensors collide


def test_build_phi_grid_floor_and_bumps():
    """phi = floor + Gaussian bumps: peaks near estimate means, ~floor far away."""
    pts, ca = make_grid((0.0, 10.0, 0.0, 10.0), 100, 100)
    centers = np.array([[3.0, 3.0], [7.0, 7.0]])
    weights = np.array([1.0, 0.5])
    phi = np.asarray(build_phi_grid(pts, centers, weights, sigma=1.0, floor=1e-3))
    pts_np = np.asarray(pts)

    def phi_at(xy):
        k = int(np.argmin(((pts_np - np.array(xy)) ** 2).sum(1)))
        return phi[k]

    assert phi_at([3.0, 3.0]) > 0.5                    # strong bump
    assert phi_at([7.0, 7.0]) > 0.25                   # weaker bump (w=0.5)
    assert phi_at([0.2, 9.8]) < 1e-2                    # far corner ~ floor
    # empty centers -> uniform floor everywhere
    phi0 = np.asarray(build_phi_grid(pts, np.zeros((0, 2)), np.zeros((0,)), 1.0, 1e-3))
    assert np.allclose(phi0, 1e-3)
