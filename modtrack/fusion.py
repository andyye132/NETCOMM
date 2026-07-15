"""Precision-weighted multi-view fusion (ModTrack Stage 4, paper Eqs. 5-8).

Given M calibrated detections {(z_m, R_m)} of the SAME target from different
sensors, fuse them by inverse-covariance (precision) weighting:

    P_indep^{-1} = sum_m R_indep,m^{-1}
    z_hat        = P_indep sum_m R_indep,m^{-1} z_m
    R_fused      = P_indep + R_pose

Each R_m is decomposed into an independent perceptual term and a shared
common-mode calibration/ego-pose term: R_m = R_indep,m + R_pose. Only the
independent part shrinks as sensors are added; the shared R_pose is added back
once so it does not spuriously vanish. Appearance features are confidence-pooled
and L2-normalized (Eq. 8).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from . import linalg
from .types import Detection, FusedDetection


def precision_weighted_fuse(
    detections: Sequence[Detection],
    R_pose: Optional[np.ndarray] = None,
) -> FusedDetection:
    """Fuse a cluster of detections of one target into a single (z_hat, R_hat).

    Parameters
    ----------
    detections : sequence of Detection
        All assumed to observe the same target (clustering happens upstream).
    R_pose : (d, d) array, optional
        Shared common-mode covariance present in every detection's R. Defaults
        to zero (all uncertainty treated as independent). Must be PSD and satisfy
        R_pose <= R (Loewner order) for every detection, so each independent term
        R - R_pose stays a valid covariance; a ValueError is raised otherwise.
    """
    n = len(detections)
    if n == 0:
        raise ValueError("cannot fuse an empty detection set")

    d = detections[0].dim
    for det in detections:
        if det.dim != d:
            raise ValueError("all detections must share the same dimension")

    Rp = np.zeros((d, d)) if R_pose is None else np.asarray(R_pose, dtype=float)
    eye = np.eye(d)

    info = np.zeros((d, d))      # sum of independent precisions  (P_indep^{-1})
    info_z = np.zeros(d)         # sum of precision-weighted means
    for det in detections:
        R_indep = det.R - Rp
        if not linalg.is_psd(R_indep):
            raise ValueError(
                "R_pose must satisfy R_pose <= R (Loewner order) for every "
                "detection; got R - R_pose that is not positive semidefinite."
            )
        # why: solve(A, I) computes A^{-1} with better conditioning than inv(A).
        precision = np.linalg.solve(R_indep, eye)
        info += precision
        info_z += precision @ det.z

    P_indep = np.linalg.solve(info, eye)
    z_hat = P_indep @ info_z
    R_fused = P_indep + Rp

    feature = _pool_features(detections)
    conf = max(det.conf for det in detections)
    members = [det.sensor_id for det in detections]
    return FusedDetection(z=z_hat, R=R_fused, feature=feature, conf=conf, members=members)


def _pool_features(detections: Sequence[Detection]) -> Optional[np.ndarray]:
    # why: confidence-weighted average of L2-normalized appearance features,
    # then re-normalize (paper Eq. 8). Returns None if no detection carries one.
    contributing = [(det.conf, det.feature) for det in detections if det.feature is not None]
    if not contributing:
        return None
    acc = None
    for beta, f in contributing:
        f_hat = f / (np.linalg.norm(f) + 1e-12)
        term = beta * f_hat
        acc = term if acc is None else acc + term
    norm = float(np.linalg.norm(acc))
    if norm < 1e-8:
        return None   # features cancelled (or all-zero confidence): no usable cue
    return acc / norm
