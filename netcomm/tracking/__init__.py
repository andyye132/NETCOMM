"""In-sim 2D BEV target tracking: drones as downward-camera sensors + a pluggable tracker.

Wires the standalone GM-PHD filter (gmphd/) into the NETCOMM sim. The existing
drones act as downward-facing cameras that detect ground targets and emit
calibrated (z, R) detections; a toggleable tracker (none / gmphd / modtrack-later)
estimates the targets in 2D BEV. The tracker is a pure estimator and stays
separable from the network loop, so tracking and network objectives can be run
independently or jointly later.
"""
from .sensors import CameraSensorConfig, camera_measurement, in_footprint, simulate_detections
from .targets import TargetConfig, TargetPopulation
from .tracker import Tracker, GMPHDTracker, make_tracker
from .repositioner import (
    Repositioner, VoronoiRepositioner, InfomaxRepositioner, MinimaxRepositioner,
    make_repositioner,
)
from .runner import run_tracking_episode, run_placed_tracking, run_preset_tracking
from . import paths

__all__ = [
    "CameraSensorConfig",
    "camera_measurement",
    "in_footprint",
    "simulate_detections",
    "TargetConfig",
    "TargetPopulation",
    "Tracker",
    "GMPHDTracker",
    "make_tracker",
    "Repositioner",
    "VoronoiRepositioner",
    "InfomaxRepositioner",
    "MinimaxRepositioner",
    "make_repositioner",
    "run_tracking_episode",
    "run_placed_tracking",
    "run_preset_tracking",
    "paths",
]
