"""Standalone GM-PHD filter for 2D BEV multi-target tracking.

A clean, self-contained implementation of the Gaussian-Mixture Probability
Hypothesis Density filter (Vo & Ma, "The Gaussian Mixture Probability Hypothesis
Density Filter", IEEE T-SP 2006), written in genuine JAX (the predict / Kalman
update / mvn_pdf / prune / merge / cap kernels are jit/vmap/scan-vectorized over
fixed-capacity padded arrays in ``kernels.py``). This is the "simpler version of
ModTrack": the closed-form GM-PHD recursion with NO appearance/identity/semantics.

Tailored for this project:
  - 2D BEV constant-velocity state  m = [px, py, vx, vy]
  - position measurements z with a PER-MEASUREMENT covariance R  (uncertainty-aware)
  - selectable birth model (``GMPHDConfig.birth_mode``):
      'measurement' (default) -- a birth Gaussian seeded at each detection
      'intensity'             -- the fixed Vo & Ma spontaneous-birth GM intensity
                                 gamma_k (paper Eq. 17), independent of detections

The filter is a pure estimator: each step it ingests a set of (z, R) detections
and returns a set of target estimates, each carrying a mean, covariance, and
weight. The covariance is the tracking-quality signal that the downstream
repositioning optimization consumes. Imports nothing from netcomm or modtrack.
"""
from .types import Detection, GaussianComponent, TargetEstimate
from .config import GMPHDConfig
from .models import cv_model, measurement_matrix, mvn_pdf
from .gmphd import GMPHDFilter

__all__ = [
    "Detection",
    "GaussianComponent",
    "TargetEstimate",
    "GMPHDConfig",
    "GMPHDFilter",
    "cv_model",
    "measurement_matrix",
    "mvn_pdf",
]
