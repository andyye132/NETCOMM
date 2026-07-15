"""TRUE paper-faithful validation against Vo & Ma 2006 (verified from the PDF).

Replicates the paper's Section III.D linear-Gaussian scenario in its Example-2
form (two fixed targets, NO spawning — spawning, Eqs. 28-30, is deliberately not
implemented in this package) with the paper's exact numbers:

  region [-1000,1000]^2, Delta=1s, sigma_nu=5 (q=25), p_S=0.99, p_D=0.98,
  birth GM = 0.1*N(.;[±250,±250,0,0], diag(100,100,25,25)),
  R = (10m)^2 I_2, clutter kappa = 12.5e-6 m^-2 (~50 returns/scan),
  T=1e-5, U=4, J_max=100, extract threshold 0.5.

Three layers of assertions, strongest first:
  1. EXACT mass identities (Corollaries 1-2, Eqs. 45-46): predicted mass =
     p_S*N_{k-1} + birth mass; posterior mass = miss mass + per-measurement
     normalized masses — the latter recomputed INDEPENDENTLY in NumPy.
  2. EXACT component-count law: J_k = (|Z_k|+1) * J_{k|k-1} before management.
  3. Tracking quality on the paper scenario (seeded): cardinality hugs the true
     count and both targets are localized within the paper's CPEP radius (20m).
Plus the Table III extraction-multiplicity rule in isolation.
"""
import numpy as np
import pytest

from gmphd import GMPHDConfig, GMPHDFilter
from gmphd.types import Detection

AREA = 2000.0 * 2000.0                      # V = 4e6 m^2
LAM_C = 12.5e-6                             # clutter intensity (m^-2) -> ~50/scan
SIGMA_EPS = 10.0                            # measurement noise std (m)
R_MEAS = SIGMA_EPS ** 2 * np.eye(2)


def _paper_config() -> GMPHDConfig:
    P_gamma = np.diag([100.0, 100.0, 25.0, 25.0])
    return GMPHDConfig(
        dt=1.0, q=25.0,                     # sigma_nu = 5 m/s^2
        p_survival=0.99, p_detect=0.98,
        clutter_intensity=LAM_C,
        birth_mode="intensity",
        birth_intensity=[
            (0.1, np.array([250.0, 250.0, 0.0, 0.0]), P_gamma),
            (0.1, np.array([-250.0, -250.0, 0.0, 0.0]), P_gamma),
        ],
        prune_threshold=1e-5, merge_threshold=4.0, max_components=100,
        extract_threshold=0.5)


def _truth(n_steps):
    """Two targets born at the birth sites, straight CV lines crossing mid-run
    (the paper's Fig.1 geometry; exact velocities are not printed in the paper,
    so these are implementer-chosen to reproduce the crossing)."""
    t1 = np.array([250.0, 250.0]) + np.arange(n_steps)[:, None] * [-5.0, -5.0]
    t2 = np.array([-250.0, -250.0]) + np.arange(n_steps)[:, None] * [5.0, 5.0]
    return np.stack([t1, t2], axis=1)       # (T, 2 targets, 2)


def _measure(true_xy, rng):
    dets = []
    for p in true_xy:
        if rng.random() < 0.98:
            dets.append(Detection(z=p + rng.normal(0.0, SIGMA_EPS, 2), R=R_MEAS.copy()))
    for _ in range(rng.poisson(LAM_C * AREA)):
        zc = rng.uniform(-1000.0, 1000.0, 2)
        dets.append(Detection(z=zc, R=R_MEAS.copy()))
    return dets


def _mvn_pdf(z, mean, S):
    d = z - mean
    return float(np.exp(-0.5 * d @ np.linalg.solve(S, d))
                 / (2.0 * np.pi * np.sqrt(np.linalg.det(S))))


def test_mass_identities_and_count_law():
    """Corollary 1 (Eq.45), Corollary 2 (Eq.46), and the exact pre-management
    component-count law, checked step by step against independent NumPy math."""
    cfg = _paper_config()
    flt = GMPHDFilter(cfg)
    rng = np.random.default_rng(0)
    truth = _truth(25)
    H = flt.H

    for k in range(25):
        prev_mass = sum(c.w for c in flt.components)
        predicted = flt.predict(flt.components) + flt.birth([])
        pred_mass = sum(c.w for c in predicted)
        # Corollary 1, no spawning: N_pred = p_S * N_{k-1} + total birth mass
        assert pred_mass == pytest.approx(0.99 * prev_mass + 0.2, rel=1e-4)

        dets = _measure(truth[k], rng)
        updated = flt.update(predicted, dets)
        # exact component-count law: J_k = (|Z_k| + 1) * J_pred
        assert len(updated) == (len(dets) + 1) * len(predicted)

        # Corollary 2, independent NumPy recomputation of the posterior mass
        expected = (1.0 - 0.98) * pred_mass
        for d in dets:
            num = 0.98 * sum(
                c.w * _mvn_pdf(d.z, H @ c.m, R_MEAS + H @ c.P @ H.T)
                for c in predicted)
            expected += num / (LAM_C + num)
        assert sum(c.w for c in updated) == pytest.approx(expected, rel=2e-3)

        flt.components = flt.cap(flt.merge(flt.prune(updated)))


def test_paper_scenario_tracks_both_targets():
    """The paper's own qualitative claim, quantified at its CPEP radius (20m):
    the filter detects and holds both targets through the mid-run crossing."""
    flt = GMPHDFilter(_paper_config())
    rng = np.random.default_rng(1)
    n_steps = 80                            # crossing at k=50 (origin)
    truth = _truth(n_steps)

    card_err, hits = [], []
    for k in range(n_steps):
        ests = flt.step(_measure(truth[k], rng))
        if k < 10:                          # birth burn-in
            continue
        card_err.append(abs(flt.cardinality - 2.0))
        pos = np.array([np.asarray(e.m[:2]) for e in ests]) if ests else np.zeros((0, 2))
        for tgt in truth[k]:
            hits.append(pos.shape[0] > 0
                        and float(np.min(np.linalg.norm(pos - tgt, axis=1))) <= 20.0)

    # calibrated over seeds 0-5: mean|N-2| in [0.25, 0.36], hit rate in [0.86, 0.95]
    # (matches the paper's "occasional over/under-estimation, not very significant"
    # at ~50 clutter returns/scan); a broken recursion reads >> 1 / << 0.5.
    assert np.mean(card_err) < 0.5, f"mean |N_hat - 2| = {np.mean(card_err):.3f}"
    assert np.mean(hits) > 0.80, f"20m-hit rate = {np.mean(hits):.3f}"


def test_table_iii_extraction_multiplicity():
    """Table III: weight > 0.5 -> round(w) copies of the mean, mass split."""
    from gmphd.types import GaussianComponent
    flt = GMPHDFilter(_paper_config())
    m = np.array([1.0, 2.0, 0.0, 0.0])
    P = np.eye(4)
    two = flt.extract([GaussianComponent(w=1.9, m=m, P=P)])
    assert len(two) == 2 and all(e.w == pytest.approx(0.95) for e in two)
    one = flt.extract([GaussianComponent(w=0.6, m=m, P=P)])
    assert len(one) == 1 and one[0].w == pytest.approx(0.6)
    assert flt.extract([GaussianComponent(w=0.4, m=m, P=P)]) == []
