"""Respawn-teleport identity handling in the evaluation pipeline.

The random_walk presets respawn an exited target at the same array row (a NEW
object at a discontinuous position). These tests pin the contract that fixes the
former corruption: frames carry per-frame ``target_ids``; _true_tracks splits
tracks at id changes; the PCRLB restarts from the diffuse prior for the new
segment (no Fisher information crosses the teleport); legacy recordings without
ids at least WARN when a teleport-sized jump is present.
"""
import numpy as np
import pytest

from evaluation import evaluate, EvalConfig
from evaluation.evaluate import _true_tracks
from netcomm.tracking.sensors import CameraSensorConfig


AREA = (0.0, 300.0, 0.0, 300.0)
T, JUMP = 30, 15


def _frames_with_teleport(with_ids=True, with_estimates=True):
    """One target walking right under a static drone for frames [0, JUMP), then
    'respawning' far away (and unobserved) for frames [JUMP, T)."""
    frames = []
    for k in range(T):
        if k < JUMP:
            pos = np.array([[10.0 + k, 50.0]])
            tid = 0
        else:
            pos = np.array([[200.0 + (k - JUMP), 250.0]])
            tid = 1
        fr = {
            "t": k,
            "drones": np.array([[17.0, 50.0, 20.0]]),   # covers only the first segment
            "targets": pos,
            "detections": [],
            "estimates": ([(pos[0].copy(), np.eye(2), 1.0)]
                          if (with_estimates and k < JUMP) else []),
        }
        if with_ids:
            fr["target_ids"] = np.array([tid])
        frames.append(fr)
    return frames


def _result(frames):
    return {"frames": frames, "area_xy": AREA, "dt": 0.5,
            "sensor_cfg": CameraSensorConfig()}


def test_true_tracks_split_on_id_change():
    tracks = _true_tracks(_frames_with_teleport(), area_xy=AREA)
    assert len(tracks) == 2
    assert sorted(tracks[0]) == list(range(0, JUMP))
    assert sorted(tracks[1]) == list(range(JUMP, T))


def test_pcrlb_restarts_at_respawn():
    out = evaluate(_result(_frames_with_teleport()), cfg=EvalConfig(compute_mot=False))
    bound_mat = np.array([pt["pos_rmse_bound"] for pt in out["_bound"]["per_target"]])
    assert bound_mat.shape == (2, T)
    # segment 1 does not exist before the teleport
    assert np.isnan(bound_mat[1, JUMP - 1]) and np.isnan(bound_mat[0, JUMP])
    # observed segment settles tight; the fresh segment restarts at diffuse-prior scale
    assert bound_mat[0, JUMP - 1] < 2.0
    assert bound_mat[1, JUMP] > 5.0, \
        "respawned track must NOT inherit the old track's Fisher information"


def test_tracked_fraction_counts_only_alive_frames():
    out = evaluate(_result(_frames_with_teleport()), cfg=EvalConfig(compute_mot=False))
    # perfect estimates for the 15 observed frames, nothing for the 15 unobserved ones
    assert out["tracked_fraction"] == pytest.approx(0.5)


def test_legacy_frames_without_ids_warn_on_teleport():
    with pytest.warns(RuntimeWarning, match="respawn"):
        evaluate(_result(_frames_with_teleport(with_ids=False)),
                 cfg=EvalConfig(compute_mot=False))


def test_no_warning_without_teleport():
    frames = _frames_with_teleport(with_ids=False)[:JUMP]      # continuous prefix only
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        evaluate(_result(frames), cfg=EvalConfig(compute_mot=False))


def test_end_to_end_random_walk_records_ids():
    """Full pipeline: a respawning random_walk run must evaluate WITHOUT the
    legacy-teleport warning because the runner records target_ids."""
    from netcomm.tracking import run_preset_tracking
    res = run_preset_tracking(
        [(150.0, 150.0, 22.0)], "random_walk", n_objects=4, n_steps=40, dt=0.2,
        area_xy=AREA, sensor_cfg=CameraSensorConfig(), object_speed=12.0,
        tracker="none", repositioner="none", seed=5)
    assert all("target_ids" in fr for fr in res["frames"])
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = evaluate(res, cfg=EvalConfig(compute_mot=False))
    assert not any("respawn" in str(w.message) for w in caught), \
        "ids are recorded, so the legacy-teleport warning must not fire"
    assert out["n_targets"] >= 4                     # respawns add identity segments
