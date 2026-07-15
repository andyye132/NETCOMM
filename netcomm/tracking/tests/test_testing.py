"""Tests for the evaluation-test harness (run_test / run_batch + CSV)."""
import numpy as np

from netcomm.tracking.testing import (
    TestConfig, run_test, run_batch, run_episode, format_table, TOP_METRICS, STATS,
)


def _fast_cfg(method="isotropic_voronoi"):
    # small area so randomly-placed drones can actually track within a few steps
    return TestConfig(method=method, n_drones=3, n_targets=3, target_motion="circle",
                      n_steps=15, dt=0.2, area=(0.0, 120.0, 0.0, 120.0))


def test_run_test_aggregates_stats_over_epochs():
    res = run_test(_fast_cfg(), n_epochs=3, n_workers=1)
    assert res["n_epochs"] == 3 and not res["errors"]
    for key, _, _ in TOP_METRICS:
        assert set(res["stats"][key]) == set(STATS)            # mean/std/min/max/median present
    assert np.isfinite(res["stats"]["gospa_mean"]["mean"])
    assert res["stats"]["gospa_mean"]["max"] >= res["stats"]["gospa_mean"]["min"]


def test_run_batch_rounds_writes_csv_and_table(tmp_path):
    rows, csv_path = run_batch(["none", "isotropic_voronoi", "greedy_mi"], _fast_cfg(),
                               n_epochs=2, n_rounds=2, batch_size=1, out_dir=tmp_path, name="unit")
    assert csv_path.exists()
    assert [r["method"] for r in rows] == ["none", "isotropic_voronoi", "greedy_mi"]
    assert all(r["epochs"] == 2 and r["rounds"] == 2 and r["n"] == 4 for r in rows)  # 2 rounds x 2 epochs
    for _, disp, _ in TOP_METRICS:                      # mean+std + round-to-round std columns
        assert f"{disp}_mean" in rows[0] and f"{disp}_round_std" in rows[0]
    table = format_table(rows)
    assert "method" in table and "GOSPA" in table and "rounds" in table
    header = csv_path.read_text().splitlines()[0]
    assert "GOSPA_mean" in header and "GOSPA_round_std" in header


def test_matched_seeds_are_deterministic():
    """A method on a fixed seed is deterministic -> within a round every method is scored on
    the IDENTICAL set of scenarios (paired comparison)."""
    cfg = _fast_cfg(method="none")
    a, b = run_episode(cfg, 321), run_episode(cfg, 321)
    for key, _, _ in TOP_METRICS:
        assert np.allclose(a[key], b[key], equal_nan=True)
    c = run_episode(cfg, 322)                            # a different seed -> generally a different scenario
    assert not all(np.allclose(a[k], c[k], equal_nan=True) for k, _, _ in TOP_METRICS)
