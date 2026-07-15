"""Ground-truth evaluation metrics for the drone multi-object tracking sim.

Because the simulator generates the true target states, it can score any
(tracker, repositioner) pair against an objective answer key. Three complementary,
canonical quantities (numpy/scipy — offline analysis, distinct from the JAX sim):

  * pcrlb     - Posterior Cramer-Rao Lower Bound (Tichavsky 1998): the dynamic
                information bound = best achievable error given the drone geometry +
                target motion (velocity/acceleration via the CV/CA model).
  * gospa     - GOSPA (alpha=2): localization / missed / false error vs truth.
  * ospa2     - OSPA^2 (Beard-Vo-Vo 2017): track-level (dynamic) error over windows.

``evaluate(result, sensor_cfg)`` ties them together on a recorded run.
"""
from .config import EvalConfig
from .gospa import gospa
from .pcrlb import pcrlb_track, pcrlb_all_targets
from .ospa2 import stitch_estimate_tracks, ospa2
from .mot import clear_mot_id_metrics
from .hota import hota
from .motion import cv_matrices, ca_matrices, model_matrices
from .evaluate import evaluate

__all__ = [
    "EvalConfig",
    "gospa",
    "pcrlb_track",
    "pcrlb_all_targets",
    "stitch_estimate_tracks",
    "ospa2",
    "clear_mot_id_metrics",
    "hota",
    "cv_matrices",
    "ca_matrices",
    "model_matrices",
    "evaluate",
]
