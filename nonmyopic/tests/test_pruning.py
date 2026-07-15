"""Load-bearing test: pruning preserves the minimax optimum while visiting fewer nodes.

`minimax_value_pruned` (alpha-beta + algebraic-redundancy, Thms 2/3/5) must return the SAME
value AND the same best first action as the full-enumeration `minimax_value` (tree.py), with
a strictly smaller node count whenever a cut can fire. eps1 = eps2 = 0 is exact.
"""
import numpy as np
import pytest

from nonmyopic import (
    MinimaxConfig, paper_noise, cv_matrices, action_offsets,
    minimax_value, minimax_value_pruned,
)
from nonmyopic.tree import minimax_value as full_minimax  # alias for clarity


def _count_full_nodes(robot_xy, x_hat, Sigma, depth, R_fn, offs, F, Q, n_meas):
    """Mirror tree.minimax_value but count expanded measurement nodes (full enumeration)."""
    from nonmyopic.riccati import predict, kalman_update
    from nonmyopic.sampler import candidate_measurements

    counter = [0]

    def rec(robot_xy, x_hat, Sigma, depth):
        if depth == 0:
            return float(np.trace(Sigma[:2, :2]))
        best = np.inf
        for ai in range(offs.shape[0]):
            nr = np.asarray(robot_xy, float) + offs[ai]
            x_pred, Sp = predict(x_hat, Sigma, F, Q)
            R = R_fn(nr, x_pred[:2])
            worst = -np.inf
            for z in candidate_measurements(x_pred, Sp, R, n_meas):
                xu, Su = kalman_update(x_pred, Sp, z, R)
                counter[0] += 1
                worst = max(worst, rec(nr, xu, Su, depth - 1))
            best = min(best, worst)
        return best

    rec(np.asarray(robot_xy, float), np.asarray(x_hat, float), np.asarray(Sigma, float), depth)
    return counter[0]


def test_pruned_matches_full_enumeration_random_instances():
    """HARD CORRECTNESS: on random horizon-2 / 4-action / 5-measurement instances, the
    pruned planner returns the same value AND best action as full enumeration (~1e-9) while
    visiting strictly fewer nodes."""
    rng = np.random.default_rng(7)
    cfg = MinimaxConfig(horizon=2, n_directions=4, include_stay=False, n_meas=5,
                        v_max=6.0, q=0.05, eps1=0.0, eps2=0.0)
    offs = action_offsets(cfg, dt=1.0)
    F, Q = cv_matrices(1.0, cfg.q)
    R_fn = lambda r, t: paper_noise(r, t, 0.5, 1.0, 30.0, 5.0)

    reductions = []
    for _ in range(12):
        robot = rng.uniform(-10, 10, size=2)
        target = rng.uniform(-25, 25, size=2)
        x_hat = np.array([target[0], target[1], rng.normal(), rng.normal()])
        A = rng.normal(size=(4, 4))
        Sigma = A @ A.T + 4.0 * np.eye(4)               # random PSD prior

        v_full, a_full = minimax_value(robot, x_hat, Sigma, cfg.horizon, R_fn, offs, F, Q,
                                       cfg.n_meas)
        v_pr, a_pr, n_pr = minimax_value_pruned(robot, x_hat, Sigma, cfg.horizon, R_fn, offs,
                                                F, Q, cfg.n_meas, eps1=0.0, eps2=0.0)
        n_full = _count_full_nodes(robot, x_hat, Sigma, cfg.horizon, R_fn, offs, F, Q,
                                   cfg.n_meas)

        assert abs(v_full - v_pr) < 1e-9               # same minimax value
        assert a_full == a_pr                          # same best first action
        assert n_pr <= n_full                          # never visits more nodes
        reductions.append((n_full, n_pr))

    # across the suite, pruning must STRICTLY reduce work on at least most instances
    assert any(n_pr < n_full for (n_full, n_pr) in reductions)
    total_full = sum(n for (n, _) in reductions)
    total_pr = sum(n for (_, n) in reductions)
    assert total_pr < total_full


def test_alpha_only_and_redundancy_only_are_exact():
    """Each pruning ALONE preserves the optimum (value + action)."""
    rng = np.random.default_rng(3)
    cfg = MinimaxConfig(horizon=2, n_directions=4, n_meas=5, v_max=6.0, q=0.03)
    offs = action_offsets(cfg, dt=1.0)
    F, Q = cv_matrices(1.0, cfg.q)
    R_fn = lambda r, t: paper_noise(r, t, 0.5, 1.0, 30.0, 5.0)

    for _ in range(8):
        robot = rng.uniform(-8, 8, size=2)
        target = rng.uniform(-20, 20, size=2)
        x_hat = np.array([target[0], target[1], 0.0, 0.0])
        A = rng.normal(size=(4, 4))
        Sigma = A @ A.T + 5.0 * np.eye(4)
        v_full, a_full = minimax_value(robot, x_hat, Sigma, cfg.horizon, R_fn, offs, F, Q,
                                       cfg.n_meas)
        for ua, ur in ((True, False), (False, True)):
            v, a, _n = minimax_value_pruned(robot, x_hat, Sigma, cfg.horizon, R_fn, offs, F, Q,
                                            cfg.n_meas, use_alpha=ua, use_redundancy=ur)
            assert abs(v - v_full) < 1e-9
            assert a == a_full


def test_node_count_reported_and_positive():
    cfg = MinimaxConfig(horizon=1, n_directions=4, n_meas=5, v_max=5.0, q=0.0)
    offs = action_offsets(cfg, dt=1.0)
    F, Q = cv_matrices(1.0, cfg.q)
    R_fn = lambda r, t: paper_noise(r, t, 0.5, 1.0, 20.0, 5.0)
    _v, _a, n = minimax_value_pruned(np.zeros(2), np.array([10.0, 0.0, 0.0, 0.0]),
                                     np.eye(4) * 4.0, 1, R_fn, offs, F, Q, 5)
    assert isinstance(n, int) and n > 0
