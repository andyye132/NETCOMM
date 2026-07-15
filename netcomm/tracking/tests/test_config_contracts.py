"""Config-provenance contracts flagged by the 2026-07-08 adversarial review.

1. make_repositioner must REJECT a config object of the wrong type instead of
   silently substituting defaults (which would make a sweep report default
   numbers under the caller's intended tuned config).
2. evaluate() must keep the PCRLB q in lockstep with the run's recorded GM-PHD
   q: auto-adopt it when no EvalConfig is passed, warn on an explicit mismatch.
"""
import numpy as np
import pytest

from coverage_control import CoverageConfig
from infomax import InfomaxConfig
from nonmyopic import MinimaxConfig
from netcomm.tracking.repositioner import make_repositioner
from netcomm.tracking.sensors import CameraSensorConfig


def test_wrong_config_type_raises():
    sensor = CameraSensorConfig()
    with pytest.raises(TypeError, match="greedy_mi.*InfomaxConfig.*CoverageConfig"):
        make_repositioner("greedy_mi", CoverageConfig(), sensor)
    with pytest.raises(TypeError, match="isotropic_voronoi.*CoverageConfig"):
        make_repositioner("isotropic_voronoi", InfomaxConfig(), sensor)
    with pytest.raises(TypeError, match="minimax.*MinimaxConfig"):
        make_repositioner("minimax", CoverageConfig(), sensor)


def test_none_and_correct_types_still_work():
    sensor = CameraSensorConfig()
    assert make_repositioner("greedy_mi", None, sensor).cfg.method == "greedy"
    assert make_repositioner("rsp", InfomaxConfig(n_d=3), sensor).cfg.n_d == 3
    assert make_repositioner("isotropic_voronoi", CoverageConfig(grid_res=32)).cfg.grid_res == 32
    assert make_repositioner("minimax", MinimaxConfig(horizon=1), sensor).cfg.horizon == 1
    assert make_repositioner("none") is None


def _tiny_run(gmphd_q):
    from gmphd import GMPHDConfig
    from netcomm.tracking import run_placed_tracking
    return run_placed_tracking(
        [(50.0, 50.0, 25.0)], [(52.0, 50.0), (46.0, 55.0)],
        n_steps=8, dt=0.5, area_xy=(0.0, 100.0, 0.0, 100.0),
        sensor_cfg=CameraSensorConfig(), gmphd_cfg=GMPHDConfig(dt=0.5, q=gmphd_q),
        tracker="gmphd", repositioner="none", seed=0)


def test_evaluate_adopts_recorded_filter_q():
    from evaluation import evaluate
    res = _tiny_run(gmphd_q=0.5)
    assert res["gmphd_cfg"].q == 0.5
    out_default = evaluate(res)                              # cfg=None -> adopts q=0.5
    from evaluation import EvalConfig
    out_matched = evaluate(res, cfg=EvalConfig(q=0.5))
    assert np.isclose(out_default["summary"]["bound_rmse_mean"],
                      out_matched["summary"]["bound_rmse_mean"])


def test_evaluate_warns_on_explicit_q_mismatch():
    from evaluation import evaluate, EvalConfig
    res = _tiny_run(gmphd_q=0.5)
    with pytest.warns(RuntimeWarning, match="GMPHDConfig.q"):
        evaluate(res, cfg=EvalConfig(q=0.01, compute_mot=False))
