"""Core math tests for the greedy/RSP MI planner: the Gaussian-MI objective is
normalized/monotone/submodular, greedy is >= 1/2 optimal, and RSP(n_d=n_r) == greedy."""
import jax
jax.config.update("jax_enable_x64", True)          # tight numerics for the math checks

import itertools

import numpy as np
import pytest

from infomax import cv_matrices, set_objective, sequential_greedy, rsp


# F = I, Q = 0 (no horizon prediction) -> myopic single-step MI for clean checks
F0, Q0 = cv_matrices(0.0, 0.0)


def _psd2(rng, scale=1.0):
    A = rng.normal(size=(2, 2))
    return scale * (A @ A.T + 0.5 * np.eye(2))


def _rand_priors(M, rng):
    P = np.zeros((M, 4, 4))
    for j in range(M):
        A = rng.normal(size=(4, 4))
        P[j] = A @ A.T + 4.0 * np.eye(4)
    return P


def _rand_candidates(n_r, n_actions, M, rng, cover_prob=0.7):
    """Per-drone (n_actions, M, 1, 2, 2) info; each action covers a random subset of targets."""
    cands = []
    for _ in range(n_r):
        c = np.zeros((n_actions, M, 1, 2, 2))
        for a in range(n_actions):
            for j in range(M):
                if rng.random() < cover_prob:
                    c[a, j, 0] = _psd2(rng, scale=rng.uniform(0.3, 1.5))
        cands.append(c)
    return cands


def test_objective_normalized():
    rng = np.random.default_rng(0)
    P = _rand_priors(3, rng)
    info = np.zeros((3, 1, 2, 2))
    assert abs(set_objective(P, np.ones(3), info, F0, Q0)) < 1e-12


def test_closed_form_mi():
    # 1 target, P0 = I4, one measurement info M = I2, no prediction:
    # MI = 1/2 logdet(I2 + P_pos * M) = 1/2 ln(det(2*I2)) = 1/2 ln(4) = ln(2) nats
    P = np.eye(4)[None]
    info = np.eye(2).reshape(1, 1, 2, 2)
    mi = set_objective(P, np.ones(1), info, F0, Q0)
    assert abs(mi - np.log(2.0)) < 1e-9


def test_objective_monotone_and_submodular():
    rng = np.random.default_rng(1)
    M = 3
    P = _rand_priors(M, rng)
    w = np.ones(M)
    base = np.zeros((M, 1, 2, 2))
    for _ in range(50):
        # nested sets B subset A via accumulating PSD info; extra element C
        B = base + sum(_psd2(rng).reshape(1, 2, 2) * (rng.random(M) < 0.6)[:, None, None]
                       for _ in range(1)).reshape(M, 1, 2, 2) * 0  # start empty-ish
        B = np.array([[_psd2(rng, 0.4) if rng.random() < 0.5 else np.zeros((2, 2))]
                      for _ in range(M)])
        extra = np.array([[_psd2(rng, 0.6)] for _ in range(M)])     # (M,1,2,2)
        A = B + np.array([[_psd2(rng, 0.5) if rng.random() < 0.5 else np.zeros((2, 2))]
                          for _ in range(M)])
        gB = set_objective(P, w, B, F0, Q0)
        gA = set_objective(P, w, A, F0, Q0)
        gBC = set_objective(P, w, B + extra, F0, Q0)
        gAC = set_objective(P, w, A + extra, F0, Q0)
        assert gA >= gB - 1e-9                                      # monotone (A >= B)
        assert (gAC - gA) <= (gBC - gB) + 1e-9                      # submodular (diminishing)


def test_greedy_at_least_half_optimal():
    rng = np.random.default_rng(2)
    n_r, n_actions, M = 3, 3, 2
    P = _rand_priors(M, rng)
    w = np.ones(M)
    cands = _rand_candidates(n_r, n_actions, M, rng)
    g = sequential_greedy(cands, P, w, F0, Q0)["total_mi"]
    # brute-force optimum over the partition matroid (one action per drone)
    best = max(set_objective(P, w, sum(cands[i][a[i]] for i in range(n_r)), F0, Q0)
               for a in itertools.product(range(n_actions), repeat=n_r))
    assert g >= 0.5 * best - 1e-9
    assert g <= best + 1e-9


def test_rsp_full_rounds_equals_greedy():
    rng = np.random.default_rng(3)
    n_r, n_actions, M = 5, 4, 3
    P = _rand_priors(M, rng)
    w = np.ones(M)
    cands = _rand_candidates(n_r, n_actions, M, rng)
    r = rsp(cands, P, w, F0, Q0, n_d=n_r, seed=7)          # one drone per round
    s = sequential_greedy(cands, P, w, F0, Q0, order=r["perm"])
    assert np.array_equal(r["chosen"], s["chosen"])
    assert abs(r["total_mi"] - s["total_mi"]) < 1e-9


def test_rsp_objective_increases_with_rounds():
    rng = np.random.default_rng(4)
    n_r, n_actions, M = 8, 4, 8
    P = _rand_priors(M, rng)
    w = np.ones(M)
    cands = _rand_candidates(n_r, n_actions, M, rng, cover_prob=0.9)
    # average objective over partition seeds for each n_d
    def mean_obj(n_d):
        return np.mean([rsp(cands, P, w, F0, Q0, n_d=n_d, seed=s)["total_mi"] for s in range(12)])
    o1, o2, o4, o8 = mean_obj(1), mean_obj(2), mean_obj(4), mean_obj(8)
    greedy = sequential_greedy(cands, P, w, F0, Q0)["total_mi"]
    assert o1 <= o2 + 1e-9 <= o4 + 1e-9 and o4 <= o8 + 1e-9   # non-decreasing in n_d
    assert o8 <= greedy + 1e-9 and o8 >= 0.9 * greedy          # n_d=n_r ~ greedy
    assert o2 >= o1                                            # any coordination beats parallel
