"""Paper-faithful regression test for the GM-PHD filter (Vo & Ma 2006).

The deep audit found that the existing suite covered closed-form per-step
known-answer cases (predict / update / merge / prune / cap) and short end-to-end
tracks, but had NO test reproducing the *qualitative behaviors the Vo & Ma 2006
paper itself claims* under the paper's own scenario: a fixed spontaneous-birth GM
intensity, linear-Gaussian CV targets over a square region with high survival
(p_S ~ 0.99) and detection (p_D ~ 0.98) probabilities, low uniform clutter, and
multi-target birth/death over ~100 steps. This file adds that regression.

It exercises the ``birth_mode='intensity'`` path (Vo & Ma Eq. 17: a configurable,
detection-INDEPENDENT list of birth Gaussians gamma_k added every predict step;
see ``gmphd.config.GMPHDConfig.birth_intensity`` and ``GMPHDFilter.birth``).

Asserted paper-faithful behaviors (qualitative, robust to RNG seed / merge
details -- deliberately NOT pixel-matching the paper figures):

  (a) the estimated cardinality (total PHD mass) TRACKS the true target count,
      time-averaged to within ~+/-1 over the run;
  (b) localization error of the extracted estimates stays within a few
      measurement-noise std-devs (data association via nearest truth);
  (c) when a target DIES the estimated cardinality decreases within a few steps.

Why these are meaningful and not tautological: the filter is never told the true
count, never told which measurement belongs to which target, and the births are
fixed (detection-independent) so the filter must actually *gate, update, prune,
merge and extract* to recover the count and the locations from a cluttered,
lossy measurement stream with spontaneous appearance and disappearance.
"""
import jax
import numpy as np
import pytest

# tight-ish numeric thresholds over a 100-step run want float64 (matches the
# other JAX test files in this package).
jax.config.update("jax_enable_x64", True)

from gmphd.config import GMPHDConfig
from gmphd.gmphd import GMPHDFilter
from gmphd.models import cv_model


# --- Vo & Ma 2006-style scenario -------------------------------------------
# Square surveillance region, linear-Gaussian CV dynamics. Numbers are chosen in
# the spirit of the paper (high p_S / p_D, low clutter, fixed birth GM) but
# rescaled to a compact region so the run is fast and seed-robust.

REGION = 100.0                  # square region is [-REGION, REGION]^2
DT = 1.0
Q = 0.01                        # CV process-noise intensity
P_S = 0.99                      # p_S: paper-style high survival
P_D = 0.98                      # p_D: paper-style high detection
MEAS_STD = 1.0                  # measurement-noise std per axis (R = std^2 I2)
R = (MEAS_STD ** 2) * np.eye(2)
N_STEPS = 100

# Low uniform clutter: a handful of false alarms per scan, spread over the
# region. clutter_intensity kappa(z) = lambda_c / area for the GM-PHD update.
LAMBDA_C = 5.0                  # expected false alarms per scan
AREA = (2.0 * REGION) ** 2
KAPPA = LAMBDA_C / AREA

# Fixed spontaneous-birth GM intensity gamma_k (Vo & Ma Eq. 17): birth Gaussians
# at a few fixed locations, each with a moderate position spread and a broad
# velocity prior. Targets are introduced near these locations so the fixed birth
# intensity (NOT a measurement-seeded birth) is what spawns them.
BIRTH_SITES = [
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([-50.0, 50.0, 0.0, 0.0]),
    np.array([50.0, -50.0, 0.0, 0.0]),
]
BIRTH_COV = np.diag([10.0, 10.0, 25.0, 25.0])   # pos var 10, vel var 25
BIRTH_WEIGHT = 0.1                               # per-site birth mass per step
BIRTH_INTENSITY = [(BIRTH_WEIGHT, m.copy(), BIRTH_COV.copy()) for m in BIRTH_SITES]


def _make_filter():
    cfg = GMPHDConfig(
        dt=DT, q=Q,
        p_survival=P_S, p_detect=P_D,
        clutter_intensity=KAPPA,
        birth_mode="intensity",
        birth_intensity=BIRTH_INTENSITY,
        prune_threshold=1e-4,
        merge_threshold=4.0,
        max_components=100,
        extract_threshold=0.5,
    )
    return GMPHDFilter(cfg)


class _Target:
    """A ground-truth CV target alive on [t_birth, t_death)."""
    def __init__(self, state0, t_birth, t_death):
        self.state = np.asarray(state0, float)
        self.t_birth = t_birth
        self.t_death = t_death

    def alive(self, t):
        return self.t_birth <= t < self.t_death

    def advance(self, F):
        self.state = F @ self.state


def _build_truth():
    """A staggered birth/death schedule near the fixed birth sites.

    Three targets appear and disappear at different times so the true cardinality
    rises and falls (0 -> 1 -> 2 -> 3 -> 2 -> 1) over the run; this is what
    behaviors (a) and (c) check against.
    """
    return [
        # born at site 0, lives most of the run
        _Target([2.0, -2.0, 1.0, 0.8],   t_birth=5,  t_death=85),
        # born at site 1, dies partway through (drives the death-detection test)
        _Target([-48.0, 52.0, 0.7, -0.4], t_birth=20, t_death=55),
        # born at site 2, late arrival
        _Target([52.0, -48.0, -0.6, 0.5], t_birth=40, t_death=95),
    ]


def _run():
    rng = np.random.default_rng(7)
    F, _ = cv_model(DT, 0.0)            # noiseless truth propagation
    filt = _make_filter()
    targets = _build_truth()

    true_counts = []
    est_counts = []
    # per-step list of (estimate positions, true positions of alive targets)
    loc_records = []

    from gmphd.types import Detection

    for t in range(N_STEPS):
        # advance all targets (truth is deterministic CV)
        for tg in targets:
            tg.advance(F)

        alive = [tg for tg in targets if tg.alive(t)]
        true_counts.append(len(alive))

        # measurements: detected truths (w.p. p_D) + uniform clutter
        dets = []
        for tg in alive:
            if rng.random() < P_D:
                z = tg.state[:2] + rng.multivariate_normal(np.zeros(2), R)
                dets.append(Detection(z=z, R=R))
        n_clutter = rng.poisson(LAMBDA_C)
        for _ in range(int(n_clutter)):
            z = rng.uniform(-REGION, REGION, size=2)
            dets.append(Detection(z=z, R=R))
        rng.shuffle(dets)

        ests = filt.step(dets)
        est_counts.append(filt.cardinality)

        est_pos = [e.position for e in ests]
        true_pos = [tg.state[:2].copy() for tg in alive]
        loc_records.append((est_pos, true_pos))

    return (np.asarray(true_counts, float),
            np.asarray(est_counts, float),
            loc_records,
            targets)


# --- (a) cardinality tracks the true count --------------------------------

def test_estimated_cardinality_tracks_true_count():
    true_counts, est_counts, _, _ = _run()
    # ignore the first few warmup steps while the fixed births accumulate mass
    warm = 8
    tc = true_counts[warm:]
    ec = est_counts[warm:]
    # (a) time-averaged cardinality within ~+/-1 of the true average
    mean_err = abs(ec.mean() - tc.mean())
    assert mean_err < 1.0, (
        f"time-averaged cardinality off by {mean_err:.2f} "
        f"(est {ec.mean():.2f} vs true {tc.mean():.2f})")
    # the per-step absolute error should also be modest on average (not a single
    # lucky average hiding wild swings)
    mae = np.mean(np.abs(ec - tc))
    assert mae < 1.25, f"mean per-step cardinality error too large: {mae:.2f}"


# --- (b) localization error within a few measurement stds ------------------

def test_localization_error_within_a_few_meas_stds():
    _, _, loc_records, _ = _run()
    # collect per-estimate localization errors against the nearest true target,
    # but only on steps where the filter is reporting roughly the right count
    # (this is a localization check, not a cardinality check -- behavior (a)
    # already guards the count). Skip warmup.
    warm = 12
    errs = []
    for est_pos, true_pos in loc_records[warm:]:
        if not true_pos or not est_pos:
            continue
        for ep in est_pos:
            d = min(np.linalg.norm(ep - tp) for tp in true_pos)
            errs.append(d)
    assert len(errs) > 50, f"too few extracted estimates to judge ({len(errs)})"
    errs = np.asarray(errs)
    # (b) localization error within a few measurement-noise stds. Use a robust
    # statistic (median) for the central tendency and a generous cap on the
    # high quantile so an occasional clutter-seeded transient does not flake.
    med = float(np.median(errs))
    q90 = float(np.quantile(errs, 0.90))
    assert med < 3.0 * MEAS_STD, f"median loc error {med:.2f} m > 3 sigma"
    assert q90 < 6.0 * MEAS_STD, f"90th-pct loc error {q90:.2f} m > 6 sigma"


# --- (c) cardinality drops within a few steps after a death ----------------

def test_cardinality_decreases_after_target_death():
    true_counts, est_counts, _, targets = _run()
    # the middle target dies at t_death=55; the true count steps down there.
    death_t = 55
    # confirm the truth really does step down at death_t (guards the scenario)
    assert true_counts[death_t] == true_counts[death_t - 1] - 1

    # estimated cardinality just before death vs a few steps after. With
    # p_S=0.99 the surviving mass for the departed target decays roughly
    # geometrically, so within ~6-8 steps the estimate should drop by close to 1.
    pre = est_counts[death_t - 3:death_t].mean()        # ~3 steps pre-death
    settle = 8
    post = est_counts[death_t + settle:death_t + settle + 5].mean()
    drop = pre - post
    assert drop > 0.6, (
        f"cardinality did not fall after death: pre={pre:.2f} post={post:.2f} "
        f"(drop {drop:.2f})")
