"""Known-answer + invariant tests for chi^2 graph clustering (Stage 3)."""
import numpy as np
import pytest

from modtrack.types import Detection
from modtrack import clustering as C
from modtrack.fusion import precision_weighted_fuse


def _det(z, R=None, sid=0, conf=1.0):
    R = np.eye(len(z)) if R is None else np.asarray(R, float)
    return Detection(z=np.asarray(z, float), R=R, sensor_id=sid, conf=conf)


# --- the chi^2 statistics --------------------------------------------------

def test_chi2_gate_value_matches_paper_table6():
    assert C.chi2_gate_value(2, 0.99) == pytest.approx(9.2103, abs=1e-3)    # camera gate
    assert C.chi2_gate_value(2, 0.999) == pytest.approx(13.816, abs=1e-3)   # radar gate
    assert C.chi2_gate_value(3, 0.99) == pytest.approx(11.345, abs=1e-3)    # 3D-ready


def test_chi2_distance_known():
    # (0,0) and (2,0), both R=I, distinct sensors -> d^2 = 4 / 2 = 2.0
    di, dj = _det([0.0, 0.0], sid=0), _det([2.0, 0.0], sid=1)
    assert C.chi2_distance(di, dj) == pytest.approx(2.0)


def test_same_target_probability_closed_form():
    # chi^2 with 2 DOF: survival = exp(-d2 / 2)
    assert C.same_target_probability(0.0, 2) == pytest.approx(1.0)
    assert C.same_target_probability(2.0, 2) == pytest.approx(np.exp(-1.0))
    # at the 99% gate the consistency score is ~1% (=1-0.99)
    assert C.same_target_probability(9.2103, 2) == pytest.approx(0.01, abs=1e-3)


# --- linking rules ---------------------------------------------------------

def test_two_consistent_detections_form_one_cluster():
    dets = [_det([0.0, 0.0], sid=0), _det([0.3, 0.0], sid=1)]   # 0.3 m apart
    clusters = C.cluster_detections(dets, tau_euc=0.5)
    assert len(clusters) == 1
    assert {d.sensor_id for d in clusters[0]} == {0, 1}


def test_euclidean_cutoff_blocks_far_pair():
    # chi^2 would pass (d^2=2 < 9.21) but 2 m apart exceeds tau_euc=0.5 -> no link
    dets = [_det([0.0, 0.0], sid=0), _det([2.0, 0.0], sid=1)]
    assert C.cluster_detections(dets, tau_euc=0.5) == []


def test_tight_covariance_fails_chi2_gate():
    # 0.4 m apart (within tau_euc) but very confident R -> d^2 = 0.16/0.002 = 80
    dets = [_det([0.0, 0.0], 0.001 * np.eye(2), sid=0),
            _det([0.4, 0.0], 0.001 * np.eye(2), sid=1)]
    assert C.cluster_detections(dets, tau_euc=0.5) == []


def test_same_sensor_detections_are_never_linked():
    # two detections at the same spot but from the SAME sensor -> no cross-sensor
    # edge -> no >=2-sensor cluster
    dets = [_det([0.0, 0.0], sid=5), _det([0.0, 0.0], sid=5)]
    assert C.cluster_detections(dets) == []


def test_transitive_closure_over_three_sensors():
    # A-B and B-C are within tau_euc, A-C is not; transitivity still merges all.
    dets = [_det([0.0, 0.0], sid=0), _det([0.3, 0.0], sid=1), _det([0.6, 0.0], sid=2)]
    clusters = C.cluster_detections(dets, tau_euc=0.5)
    assert len(clusters) == 1
    assert {d.sensor_id for d in clusters[0]} == {0, 1, 2}


def test_sensor_uniqueness_keeps_highest_confidence():
    # sensor 0 appears twice; only its higher-confidence detection survives.
    dets = [
        _det([0.00, 0.0], sid=0, conf=0.9),
        _det([0.10, 0.0], sid=0, conf=0.5),   # same sensor, lower conf -> dropped
        _det([0.05, 0.0], sid=1, conf=0.8),
    ]
    clusters = C.cluster_detections(dets, tau_euc=0.5)
    assert len(clusters) == 1
    cl = clusters[0]
    assert sorted(d.sensor_id for d in cl) == [0, 1]
    s0 = next(d for d in cl if d.sensor_id == 0)
    assert s0.conf == pytest.approx(0.9)               # kept the conf=0.9 one
    np.testing.assert_allclose(s0.z, [0.0, 0.0])


# --- component filtering ---------------------------------------------------

def test_singleton_is_discarded_by_default():
    assert C.cluster_detections([_det([0.0, 0.0], sid=0)]) == []


def test_min_sensors_one_keeps_singletons():
    clusters = C.cluster_detections([_det([1.0, 2.0], sid=0)], min_sensors=1)
    assert len(clusters) == 1 and len(clusters[0]) == 1


def test_two_separate_targets_yield_two_clusters():
    dets = [
        _det([0.0, 0.0], sid=0), _det([0.2, 0.0], sid=1),     # target A
        _det([9.0, 9.0], sid=0), _det([9.1, 9.0], sid=1),     # target B, far away
    ]
    clusters = C.cluster_detections(dets, tau_euc=0.5)
    assert len(clusters) == 2
    centers = sorted(np.mean([d.z for d in cl], axis=0)[0] for cl in clusters)
    assert centers[0] < 1.0 and centers[1] > 8.0


# --- confidence cascade ----------------------------------------------------

def test_cascade_low_confidence_attaches_to_high():
    dets = [
        _det([0.00, 0.0], sid=0, conf=0.9),   # high
        _det([0.10, 0.0], sid=1, conf=0.9),   # high
        _det([0.05, 0.0], sid=2, conf=0.3),   # low -> should attach
    ]
    clusters = C.cluster_detections(dets, tau_euc=0.5, tau_high=0.5, tau_low=0.1)
    assert len(clusters) == 1
    assert {d.sensor_id for d in clusters[0]} == {0, 1, 2}


def test_cascade_isolated_low_confidence_is_dropped():
    dets = [
        _det([0.00, 0.0], sid=0, conf=0.9),
        _det([0.10, 0.0], sid=1, conf=0.9),
        _det([9.00, 9.0], sid=2, conf=0.3),   # low and far from any high -> dropped
    ]
    clusters = C.cluster_detections(dets, tau_euc=0.5, tau_high=0.5, tau_low=0.1)
    assert len(clusters) == 1
    assert {d.sensor_id for d in clusters[0]} == {0, 1}


# --- end-to-end: cluster -> fuse ------------------------------------------

def test_cluster_then_fuse_pipeline():
    dets = [_det([0.0, 0.0], sid=0), _det([0.2, 0.0], sid=1)]
    clusters = C.cluster_detections(dets, tau_euc=0.5)
    fused = [precision_weighted_fuse(c) for c in clusters]
    assert len(fused) == 1
    # fused mean is the midpoint (equal R), covariance shrinks below each input
    np.testing.assert_allclose(fused[0].z, [0.1, 0.0], atol=1e-9)
    assert np.linalg.eigvalsh(fused[0].R).max() < 1.0
