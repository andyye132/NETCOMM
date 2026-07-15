"""Tests for the BEV tracking visualizer: ellipse math + render-to-file smoke."""
import numpy as np
import pytest

from netcomm.tracking.visualize import covariance_ellipse, animate_tracking, plot_frame


def test_covariance_ellipse_axis_aligned_known():
    # diag(4, 1), n_std = 1: major axis along x -> width 4, height 2, angle 0
    w, h, a = covariance_ellipse(np.diag([4.0, 1.0]), n_std=1.0)
    assert w == pytest.approx(4.0)
    assert h == pytest.approx(2.0)
    assert abs(a) < 1e-6


def test_covariance_ellipse_rotates_to_major_axis():
    # diag(1, 4): major axis along y -> width (major) 4, angle ~ 90 deg
    w, h, a = covariance_ellipse(np.diag([1.0, 4.0]), n_std=1.0)
    assert w == pytest.approx(4.0)
    assert h == pytest.approx(2.0)
    assert abs(abs(a) - 90.0) < 1e-6


def test_covariance_ellipse_confidence_scaling():
    # 95% in 2D scales by sqrt(chi2.ppf(0.95, 2)) ~= 2.4477
    w, _, _ = covariance_ellipse(np.eye(2), confidence=0.95)
    assert w == pytest.approx(2.0 * np.sqrt(5.99146), rel=1e-3)


def _synthetic_result():
    frames = []
    for t in range(4):
        frames.append({
            "t": t,
            "drones": np.array([[40.0, 50.0, 20.0], [60.0, 55.0, 20.0]]),
            "targets": np.array([[50.0 + t, 50.0], [20.0, 80.0 - t]]),
            "detections": [(np.array([50.0 + t, 50.0]), np.eye(2))],
            "estimates": [(np.array([50.0 + t, 50.0]), np.diag([1.0, 4.0]), 1.0)],
            "n_estimates": 1,
        })
    return {"frames": frames, "area_xy": (0.0, 100.0, 0.0, 100.0),
            "sensor_half_fov_rad": float(np.deg2rad(50.0)), "n_targets": 2}


def test_plot_frame_writes_png(tmp_path):
    out = tmp_path / "frame.png"
    plot_frame(_synthetic_result(), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_animate_writes_gif(tmp_path):
    out = tmp_path / "anim.gif"
    animate_tracking(_synthetic_result(), str(out), fps=4)
    assert out.exists() and out.stat().st_size > 0


def test_empty_frames_raise(tmp_path):
    with pytest.raises(ValueError):
        plot_frame({"frames": [], "area_xy": (0, 1, 0, 1)}, str(tmp_path / "x.png"))


def test_covariance_ellipse_nan_returns_no_ellipse():
    assert covariance_ellipse(np.array([[1.0, np.nan], [np.nan, 1.0]])) == (0.0, 0.0, 0.0)


def test_covariance_ellipse_rejects_non_2x2():
    with pytest.raises(ValueError):
        covariance_ellipse(np.eye(3))


def test_varying_target_count_raises(tmp_path):
    frames = [
        {"t": 0, "drones": np.zeros((1, 3)), "targets": np.array([[1.0, 1.0], [2.0, 2.0]]),
         "detections": [], "estimates": [], "n_estimates": 0},
        {"t": 1, "drones": np.zeros((1, 3)), "targets": np.array([[1.0, 1.0]]),
         "detections": [], "estimates": [], "n_estimates": 0},
    ]
    res = {"frames": frames, "area_xy": (0.0, 10.0, 0.0, 10.0), "sensor_half_fov_rad": 0.5}
    with pytest.raises(ValueError):
        animate_tracking(res, str(tmp_path / "a.gif"))
