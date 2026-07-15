"""Correctness tests for the ground-truth evaluation metrics."""
import numpy as np
import pytest

from evaluation import (
    gospa, EvalConfig, pcrlb_track, model_matrices, ospa2, stitch_estimate_tracks, evaluate,
)
from netcomm.tracking.sensors import (
    CameraSensorConfig, camera_measurement, measurement_information, los_probability,
)


# --------------------------------------------------------------------------- GOSPA
def test_gospa_perfect_match_is_zero():
    g = gospa([[0, 0], [10, 10]], [[0, 0], [10, 10]], c=5, p=2, alpha=2)
    assert g["total"] == 0.0 and g["missed"] == 0 and g["false"] == 0 and g["n_matched"] == 2


def test_gospa_alpha2_decomposition_exact_values():
    # 2 exact matches + 1 spurious estimate (false) + 1 unseen truth (missed)
    est = [[0, 0], [10, 10], [3, 3]]
    tru = [[0, 0], [10, 10], [50, 50]]
    c, p = 5.0, 2.0
    g = gospa(est, tru, c=c, p=p, alpha=2)
    assert g["false"] == 1 and g["missed"] == 1 and g["n_matched"] == 2
    # exact hand-computed magnitudes (not the tautological decomposition identity):
    assert abs(g["localization"] - 0.0) < 1e-12                  # the two matches are exact
    assert abs(g["total"] - np.sqrt((c ** p / 2.0) * 2)) < 1e-9  # = c = 5.0


def _brute_gospa(est, tru, c, p, alpha=2.0):
    """Reference GOSPA by brute force over all partial assignments (canonical capped distance)."""
    import itertools
    E = np.asarray(est, dtype=float).reshape(-1, 2)
    Tr = np.asarray(tru, dtype=float).reshape(-1, 2)
    m, n = len(E), len(Tr)
    pen = (c ** p) / alpha
    best = float("inf")
    for k in range(min(m, n) + 1):
        for esub in itertools.combinations(range(m), k):
            for tsub in itertools.permutations(range(n), k):
                loc = sum(min(float(np.linalg.norm(E[ei] - Tr[tj])), c) ** p
                          for ei, tj in zip(esub, tsub))
                best = min(best, loc + pen * ((m - k) + (n - k)))
    return best ** (1.0 / p)


def test_gospa_matches_bruteforce_oracle():
    rng = np.random.default_rng(0)
    for _ in range(40):
        m, n = rng.integers(0, 4), rng.integers(0, 4)
        est = rng.uniform(0, 10, size=(m, 2))
        tru = rng.uniform(0, 10, size=(n, 2))
        got = gospa(est, tru, c=5.0, p=2.0, alpha=2)["total"]
        ref = _brute_gospa(est, tru, c=5.0, p=2.0, alpha=2) if (m or n) else 0.0
        assert abs(got - ref) < 1e-9


def test_gospa_rejects_alpha_not_two():
    with pytest.raises(ValueError):
        gospa([[0, 0]], [[0, 0]], c=5, p=2, alpha=1.0)


def test_gospa_localization_and_cutoff():
    assert abs(gospa([[1, 0]], [[0, 0]], c=5, p=2)["total"] - 1.0) < 1e-9
    # beyond cutoff -> counts as 1 missed + 1 false, not a 100-m localization
    g = gospa([[100, 0]], [[0, 0]], c=5, p=2, alpha=2)
    assert g["n_matched"] == 0 and g["missed"] == 1 and g["false"] == 1


def test_gospa_empty_sets():
    assert gospa([], [], c=5)["total"] == 0.0
    assert gospa([], [[0, 0]], c=5, p=2, alpha=2)["missed"] == 1
    assert gospa([[0, 0]], [], c=5, p=2, alpha=2)["false"] == 1


# --------------------------------------------------------------------------- PCRLB
def test_pcrlb_equals_kalman_riccati_linear_gaussian():
    """With p_detect=1 and a single always-visible drone, the PCRLB information
    recursion is exactly the Kalman covariance Riccati: inv(J_k) == P_k."""
    cfg = CameraSensorConfig(p_detect=1.0)           # deterministic detection, no LoS/NLoS
    drone = np.array([[0.0, 0.0, 15.0]])
    truth = np.zeros((30, 2))                         # stationary target overhead -> constant R
    dt, q = 0.2, 0.05
    out = pcrlb_track(truth, [drone] * 30, cfg, motion="cv", q=q, dt=dt,
                      prior_pos_var=25.0, prior_vel_var=100.0)

    # independent Kalman covariance (Riccati) recursion with the same F, Q, H, R
    F, Q, H = model_matrices("cv", dt, q)
    _, R = camera_measurement(drone[0], truth[0], cfg)
    Rinv = np.linalg.inv(R)
    P = np.diag([25.0, 25.0, 100.0, 100.0])
    rmse_kf = []
    for _ in range(30):
        P_pred = F @ P @ F.T + Q
        P = np.linalg.inv(np.linalg.inv(P_pred) + H.T @ Rinv @ H)
        rmse_kf.append(np.sqrt(np.trace(P[:2, :2])))      # 2D radial RMS (matches pcrlb convention)
    assert np.allclose(out["pos_rmse_bound"], rmse_kf, rtol=1e-6, atol=1e-9)


def test_pcrlb_more_drones_more_information():
    cfg = CameraSensorConfig()
    truth = np.zeros((20, 2))
    one = [np.array([[0.0, 0.0, 15.0]])] * 20
    three = [np.array([[0.0, 0.0, 15.0], [3.0, 0.0, 15.0], [0.0, 3.0, 15.0]])] * 20
    b1 = pcrlb_track(truth, one, cfg, dt=0.2)
    b3 = pcrlb_track(truth, three, cfg, dt=0.2)
    assert b3["pos_rmse_bound"][-1] < b1["pos_rmse_bound"][-1]   # more drones -> tighter bound
    assert np.all(b3["info_gain"] >= 0)


def test_pcrlb_information_erodes_without_observation():
    """When no drone sees the target, the bound RMSE grows (info eroded by Q)."""
    cfg = CameraSensorConfig()
    truth = np.zeros((10, 2))
    far = [np.array([[500.0, 500.0, 15.0]])] * 10        # drone never in footprint
    b = pcrlb_track(truth, far, cfg, dt=0.2)
    assert b["n_observing"].sum() == 0
    assert b["pos_rmse_bound"][-1] > b["pos_rmse_bound"][0]     # uncertainty grows
    assert np.allclose(b["info_gain"], 0.0, atol=1e-9)         # no measurement info gained


def test_pcrlb_ca_model_runs_with_six_state():
    cfg = CameraSensorConfig()
    F, Q, H = model_matrices("ca", 0.2, 0.05)
    assert F.shape == (6, 6) and H.shape == (2, 6)
    b = pcrlb_track(np.zeros((10, 2)), [np.array([[0.0, 0.0, 15.0]])] * 10, cfg,
                    motion="ca", q=0.05, dt=0.2)
    assert np.all(np.isfinite(b["pos_rmse_bound"]))


def test_los_nlos_reduces_information_vs_clean():
    """LoS/NLoS sensing yields less expected information than the clean model at an
    oblique angle (some looks are NLoS with inflated R / lower p_D)."""
    fov = np.deg2rad(60.0)                                  # footprint ~26 m so (20,0) is in view
    clean = CameraSensorConfig(half_fov_rad=fov, los_nlos=False)
    los = CameraSensorConfig(half_fov_rad=fov, los_nlos=True)
    drone, target = np.array([0.0, 0.0, 15.0]), np.array([20.0, 0.0])   # oblique, in footprint
    info_clean = np.trace(measurement_information(drone, target, clean))
    info_los = np.trace(measurement_information(drone, target, los))
    assert info_los < info_clean and info_los > 0


# --------------------------------------------------------------------------- OSPA^2
def test_ospa2_identical_tracks_zero_and_missing_track_penalized():
    true_tracks = [{k: np.array([float(k), 0.0]) for k in range(20)},
                   {k: np.array([0.0, float(k)]) for k in range(20)}]
    # estimates identical to truth -> ~0
    s_same = ospa2([dict(t) for t in true_tracks], true_tracks, 20, c=10, p=2, window=10)
    assert np.all(s_same < 1e-9)
    # drop one estimated track -> nonzero (cardinality mismatch)
    s_miss = ospa2([dict(true_tracks[0])], true_tracks, 20, c=10, p=2, window=10)
    assert np.mean(s_miss) > 1.0


def test_ospa2_constant_offset_equals_offset():
    """One est + one true track, both present over the whole window at a constant
    offset d < c: OSPA^2 == d every window (no cardinality term)."""
    true = [{k: np.array([0.0, 0.0]) for k in range(5)}]
    est = [{k: np.array([3.0, 0.0]) for k in range(5)}]
    s = ospa2(est, true, 5, c=10.0, p=2.0, window=5)
    assert np.allclose(s, 3.0, atol=1e-9)


def test_ospa2_cardinality_mismatch_reference_value():
    """2 true tracks, 1 est track perfectly matching one of them. m=1, n=2:
    loc=0 for the match, card = c^p*(n-m); OSPA^2 = (c^p / 2)^(1/p) = c/sqrt(2)."""
    true = [{k: np.array([0.0, 0.0]) for k in range(5)},
            {k: np.array([100.0, 0.0]) for k in range(5)}]
    est = [{k: np.array([0.0, 0.0]) for k in range(5)}]
    s = ospa2(est, true, 5, c=10.0, p=2.0, window=5)
    assert abs(s[0] - 10.0 / np.sqrt(2.0)) < 1e-9          # = 7.0710678


def test_ospa2_partial_presence_reference_value():
    """One true track present all 4 frames; one est track exactly matching but present
    only the first 2 frames. Base track distance over the window is
    ((0+0+c^2+c^2)/4)^(1/2) = c/sqrt(2); single track each -> OSPA^2 = c/sqrt(2)."""
    true = [{k: np.array([0.0, 0.0]) for k in range(4)}]
    est = [{0: np.array([0.0, 0.0]), 1: np.array([0.0, 0.0])}]
    s = ospa2(est, true, 4, c=10.0, p=2.0, window=4)
    assert abs(s[0] - 10.0 / np.sqrt(2.0)) < 1e-9


def _brute_ospa2_window(A, B, window, c, p):
    """Independent OSPA^2 set distance over a single window: brute-force the optimal
    assignment via permutations (vs the code's scipy linear_sum_assignment)."""
    import itertools

    def base(a, b):
        num = cnt = 0
        for t in window:
            pa, pb = a.get(t), b.get(t)
            if pa is None and pb is None:
                continue
            d = c if (pa is None or pb is None) else min(c, float(np.linalg.norm(pa - pb)))
            num += d ** p
            cnt += 1
        return 0.0 if cnt == 0 else (num / cnt) ** (1.0 / p)

    m, n = len(A), len(B)
    if m == 0 and n == 0:
        return 0.0
    if m > n:
        A, B, m, n = B, A, n, m
    if m == 0:
        return c
    best = None
    for cols in itertools.permutations(range(n), m):
        loc = sum(min(c, base(A[i], B[cols[i]])) ** p for i in range(m))
        best = loc if best is None or loc < best else best
    card = (c ** p) * (n - m)
    return ((best + card) / n) ** (1.0 / p)


def test_ospa2_matches_bruteforce_assignment_oracle():
    """OSPA^2's scipy assignment must match an independent brute-force optimal
    assignment on random small track sets present over the full window."""
    rng = np.random.default_rng(7)
    c, p, L = 10.0, 2.0, 4
    for _ in range(40):
        m, n = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        A = [{k: rng.uniform(-8, 8, size=2) for k in range(L)} for _ in range(m)]
        B = [{k: rng.uniform(-8, 8, size=2) for k in range(L)} for _ in range(n)]
        got = ospa2(A, B, L, c=c, p=p, window=L)[0]
        ref = _brute_ospa2_window(A, B, list(range(L)), c, p)
        assert abs(got - ref) < 1e-9


def test_clear_mot_id_metrics_worked_example():
    """Validate the py-motmetrics adapter against a hand-computed scenario (5 frames 0..4).
    True tracks A, B. Estimates: id0 tracks A every frame (dist 0.1); B is tracked as id1
    on frames 0,1 (dist 0.2) then as id2 on frames 2,4 (dist 0.3) with frame 3 missed
    (one ID switch at frame 2); a far id3 at frame 4 is a false positive.
    => FN=1, FP=1, IDSW=1 -> MOTA = 1-3/10 = 0.70; MOTP = 1.5/9 = 0.16667; IDF1 = 14/20 = 0.70."""
    from evaluation import clear_mot_id_metrics
    pytest.importorskip("motmetrics")
    true_tracks = [{t: np.array([float(t), 0.0]) for t in range(5)},      # A
                   {t: np.array([0.0, float(t)]) for t in range(5)}]      # B
    est_tracks = [
        {t: np.array([float(t), 0.1]) for t in range(5)},                # id0 -> A (dist 0.1)
        {0: np.array([0.0, 0.2]), 1: np.array([0.0, 1.2])},              # id1 -> B frames 0,1
        {2: np.array([0.0, 2.3]), 4: np.array([0.0, 4.3])},             # id2 -> B frames 2,4 (3 missed)
        {4: np.array([100.0, 100.0])},                                  # id3 -> false positive
    ]
    m = clear_mot_id_metrics(est_tracks, true_tracks, n_frames=5, tau=5.0)
    assert m["misses"] == 1 and m["false_positives"] == 1 and m["id_switches"] == 1
    assert abs(m["mota"] - 0.70) < 1e-9
    assert abs(m["motp"] - 1.5 / 9.0) < 1e-6
    assert abs(m["idf1"] - 0.70) < 1e-6


def test_clear_mot_perfect_tracking():
    from evaluation import clear_mot_id_metrics
    pytest.importorskip("motmetrics")
    true_tracks = [{t: np.array([float(t), 0.0]) for t in range(6)},
                   {t: np.array([0.0, float(t)]) for t in range(6)}]
    est_tracks = [dict(true_tracks[0]), dict(true_tracks[1])]      # exact, persistent ids
    m = clear_mot_id_metrics(est_tracks, true_tracks, n_frames=6, tau=5.0)
    assert m["mota"] == 1.0 and m["idf1"] == 1.0 and m["id_switches"] == 0
    assert abs(m["motp"]) < 1e-12


def test_hota_worked_example_matches_trackeval():
    """Validate the self-contained HOTA against the TrackEval-verified worked example
    (5 frames, tau=1.0): track 1 follows A then ID-switches to B, track 2 covers B early,
    track 3 is a false positive, A is missed on frames 2-4. Loc error 0.1 -> S=0.9."""
    from evaluation import hota
    p = lambda x, y: np.array([float(x), float(y)])
    true_tracks = [{f: p(0, 0) for f in range(5)},          # A
                   {f: p(10, 0) for f in range(5)}]         # B
    est_tracks = [
        {0: p(0, 0.1), 1: p(0, 0.1), 2: p(10, 0.1), 3: p(10, 0.1), 4: p(10, 0.1)},  # 1: A then B
        {0: p(10, 0.1), 1: p(10, 0.1)},                                              # 2: B early
        {2: p(50, 50)},                                                              # 3: false positive
    ]
    r = hota(est_tracks, true_tracks, n_frames=5, tau=1.0)
    assert abs(r["hota"] - 0.45932) < 1e-4
    assert abs(r["deta"] - 0.60287) < 1e-4
    assert abs(r["assa"] - 0.34995) < 1e-4
    assert abs(r["loca"] - 0.90526) < 1e-4
    assert abs(r["hota_alpha0"] - 0.48483) < 1e-4


def test_hota_perfect_and_disjoint():
    from evaluation import hota
    tt = [{t: np.array([float(t), 0.0]) for t in range(6)},
          {t: np.array([0.0, float(t)]) for t in range(6)}]
    perfect = hota([dict(tt[0]), dict(tt[1])], tt, 6, tau=5.0)
    assert abs(perfect["hota"] - 1.0) < 1e-9 and abs(perfect["assa"] - 1.0) < 1e-9
    far = [{t: np.array([500.0, 500.0]) for t in range(6)}]      # never within tau
    disjoint = hota(far, tt, 6, tau=5.0)
    assert disjoint["hota"] == 0.0


def test_hota_id_switch_lowers_assa_not_deta():
    """A perfectly-detected track that switches ID halfway keeps DetA=1 but drops AssA."""
    from evaluation import hota
    true_tracks = [{t: np.array([float(t), 0.0]) for t in range(6)}]
    est_tracks = [{0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0]), 2: np.array([2.0, 0.0])},
                  {3: np.array([3.0, 0.0]), 4: np.array([4.0, 0.0]), 5: np.array([5.0, 0.0])}]
    r = hota(est_tracks, true_tracks, 6, tau=5.0)
    assert abs(r["deta"] - 1.0) < 1e-9          # every frame detected
    assert abs(r["assa"] - 0.5) < 1e-9          # split 3/3 -> 3/(6+3-3)=0.5 each, weighted mean 0.5
    assert abs(r["hota"] - np.sqrt(0.5)) < 1e-9


def test_stitch_recovers_two_separated_tracks():
    # two well-separated, steadily-moving estimates -> exactly two stitched tracks
    frames = []
    for k in range(15):
        frames.append({"estimates": [(np.array([float(k), 0.0]), np.eye(4), 1.0),
                                      (np.array([0.0, float(k) + 50.0]), np.eye(4), 1.0)]})
    tracks = stitch_estimate_tracks(frames, gate=5.0)
    assert len(tracks) == 2 and all(len(t) == 15 for t in tracks)


# --------------------------------------------------------------------------- end to end
def test_evaluate_end_to_end_on_placed_run():
    from netcomm.tracking import run_placed_tracking
    res = run_placed_tracking(
        [(50.0, 50.0, 18.0), (60.0, 55.0, 18.0)], [(52.0, 50.0), (58.0, 54.0)],
        n_steps=20, dt=0.2, area_xy=(0.0, 120.0, 0.0, 120.0),
        sensor_cfg=CameraSensorConfig(half_fov_rad=np.deg2rad(60.0)),
        object_speed=1.0, tracker="gmphd", seed=0)
    ev = evaluate(res, CameraSensorConfig(half_fov_rad=np.deg2rad(60.0)),
                  EvalConfig(), dt=0.2)
    s = ev["summary"]
    assert ev["n_frames"] == 20 and ev["n_targets"] == 2
    for key in ("gospa_mean", "bound_rmse_mean", "achieved_rmse_mean",
                "efficiency_mean", "efficiency_median", "info_gain_cumulative", "ospa2_mean"):
        assert key in s
    assert np.isfinite(s["gospa_mean"]) and s["bound_rmse_mean"] > 0
    # efficiency = achieved RMS / bound RMS (same 2D-radial units): finite and positive. The RMS
    # (squared) aggregate can be inflated by a rare single-frame GM-PHD under-extraction transient
    # on two converging targets, so bound it only loosely here.
    assert np.isfinite(s["efficiency_mean"]) and 0.3 < s["efficiency_mean"] < 10.0
    # the robust (median) per-frame efficiency reflects sustained tracker quality and must sit
    # near the bound (~order 1: the tracker is roughly 1-2x the PCRLB), unaffected by transients.
    assert np.isfinite(s["efficiency_median"]) and 0.3 < s["efficiency_median"] < 3.0
    assert s["tracked_fraction"] > 0.0
