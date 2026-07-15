"""In-sim smoke tests: the tracking loop runs on the NETCOMM world and toggles."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # no GPU on this machine

import jax
import pytest

import numpy as np

from experiments._runner import load_scenario_method, build_cfg
from netcomm.tracking import run_tracking_episode, run_placed_tracking, make_tracker


def _cfg():
    scn, _ = load_scenario_method("open_field", "adaptive")
    return build_cfg(scn)


def test_run_tracking_episode_smoke():
    out = run_tracking_episode(_cfg(), n_steps=12, key=jax.random.PRNGKey(0),
                               tracker="gmphd")
    assert len(out["frames"]) == 12
    assert out["tracker"] == "gmphd"
    assert "final_cardinality" in out
    # every frame carries the recorded structure
    for f in out["frames"]:
        assert {"drones", "targets", "detections", "estimates", "n_estimates"} <= set(f)


def test_tracking_toggle_off_produces_no_estimates():
    out = run_tracking_episode(_cfg(), n_steps=8, key=jax.random.PRNGKey(1),
                               tracker="none")
    assert all(f["n_estimates"] == 0 for f in out["frames"])
    assert "final_cardinality" not in out


def test_make_tracker_toggle():
    assert make_tracker("none") is None
    assert make_tracker("gmphd").name == "gmphd"
    with pytest.raises(NotImplementedError):
        make_tracker("modtrack")
    with pytest.raises(ValueError):
        make_tracker("bogus")


# --- placed-scenario backend (for the app) --------------------------------

def test_run_placed_tracking_tracks_a_placed_object():
    # three radius-30 cameras around (50,50), one object placed there
    drones = [(48.0, 50.0, 30.0), (52.0, 53.0, 30.0), (50.0, 46.0, 30.0)]
    out = run_placed_tracking(drones, [(50.0, 50.0)], n_steps=20, dt=1.0,
                              area_xy=(0, 100, 0, 100), object_speed=1.0, seed=1)
    assert len(out["frames"]) == 20
    last = out["frames"][-1]
    truth = last["targets"][0]
    ests = last["estimates"]
    assert len(ests) >= 1
    best = min(ests, key=lambda e: np.linalg.norm(e[0] - truth))
    assert np.linalg.norm(best[0] - truth) < 3.0


def test_run_placed_tracking_no_drones_no_tracks():
    out = run_placed_tracking([], [(50.0, 50.0)], n_steps=10, dt=1.0, area_xy=(0, 100, 0, 100))
    assert all(f["n_estimates"] == 0 for f in out["frames"])


def test_run_placed_tracking_no_objects():
    out = run_placed_tracking([(50.0, 50.0, 30.0)], [], n_steps=10, dt=1.0,
                              area_xy=(0, 100, 0, 100))
    assert out["n_targets"] == 0
    assert all(len(f["detections"]) == 0 and f["n_estimates"] == 0 for f in out["frames"])
