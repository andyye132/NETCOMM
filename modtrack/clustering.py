"""chi^2 graph clustering (ModTrack Stage 3, paper Section 3.3).

Group calibrated (z, R) detections from DIFFERENT sensors that correspond to the
same target, before precision-weighted fusion combines each group.

Two detections are linked by an edge when they are statistically consistent
under their combined covariance — a chi^2 gate on the Mahalanobis distance

    d^2_ij = (z_i - z_j)^T (R_i + R_j)^{-1} (z_i - z_j)   ~  chi^2(dim)

AND within a hard Euclidean cutoff ``tau_euc`` (which stops far-apart detections
from chaining together transitively through intermediates). Connected components
of this graph are the candidate clusters. We then enforce at-most-one-detection-
per-sensor (keeping higher confidence first) and discard clusters supported by
fewer than ``min_sensors`` distinct sensors before fusion.

A ByteTrack-style confidence cascade is optional: when ``tau_high`` is set,
high-confidence detections are clustered first and lower-confidence detections
(>= ``tau_low``) only attach to those high-confidence components.

All gating is dimension-general; only the gate *value* depends on the state
dimension. ``chi2_gate`` defaults to the 2D / 99% value (9.21); for 3D use
``chi2_gate_value(3, 0.99)``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2 as _chi2

from .types import Detection
from .linalg import mahalanobis_sq


def chi2_gate_value(dim: int = 2, confidence: float = 0.99) -> float:
    """chi^2 quantile for ``dim`` DOF at ``confidence`` (dim=2, 0.99 -> 9.21)."""
    return float(_chi2.ppf(confidence, dim))


def chi2_distance(det_i: Detection, det_j: Detection) -> float:
    """Mahalanobis d^2 under summed covariances: (zi-zj)^T (Ri+Rj)^-1 (zi-zj)."""
    return mahalanobis_sq(det_i.z - det_j.z, det_i.R + det_j.R)


def same_target_probability(d2: float, dim: int = 2) -> float:
    """chi^2 consistency score P_same = 1 - F_{chi^2,dim}(d2).

    For 2 DOF this has the closed form exp(-d2 / 2).
    """
    return float(_chi2.sf(d2, dim))


# ---------------------------------------------------------------------------
# Graph machinery
# ---------------------------------------------------------------------------

def _connected_components(n: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # path halving
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    comps: dict = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def _link(dets: Sequence[Detection], a: int, b: int,
          chi2_gate: float, tau_euc: float) -> bool:
    # why: only ever link DISTINCT sensors; require both the hard Euclidean cutoff
    # and the chi^2 consistency gate.
    if dets[a].sensor_id == dets[b].sensor_id:
        return False
    if float(np.linalg.norm(dets[a].z - dets[b].z)) > tau_euc:
        return False
    return chi2_distance(dets[a], dets[b]) <= chi2_gate


def _within_group_edges(dets, group, chi2_gate, tau_euc):
    edges = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            if _link(dets, a, b, chi2_gate, tau_euc):
                edges.append((a, b))
    return edges


def _cross_group_edges(dets, group, anchors, chi2_gate, tau_euc):
    edges = []
    for a in group:
        for b in anchors:
            if _link(dets, a, b, chi2_gate, tau_euc):
                edges.append((a, b))
    return edges


def _sensor_unique(component: Sequence[int], dets: Sequence[Detection]) -> List[int]:
    # why: at most one detection per sensor; keep the highest-confidence one.
    kept: dict = {}
    for i in sorted(component, key=lambda k: -dets[k].conf):
        kept.setdefault(dets[i].sensor_id, i)
    return list(kept.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_detections(
    detections: Sequence[Detection],
    chi2_gate: float = 9.21,
    tau_euc: float = 0.5,
    tau_high: Optional[float] = None,
    tau_low: Optional[float] = None,
    min_sensors: int = 2,
) -> List[List[Detection]]:
    """Cluster detections into same-target groups ready for fusion.

    Returns a list of clusters; each cluster is a list of Detections with at most
    one per sensor and at least ``min_sensors`` distinct sensors.
    """
    dets = list(detections)
    n = len(dets)
    if n == 0:
        return []

    if tau_high is None:
        # single pass over all detections
        active = list(range(n))
        edges = _within_group_edges(dets, active, chi2_gate, tau_euc)
    else:
        # ByteTrack-style cascade: high-confidence first, low-confidence attaches.
        low_floor = 0.0 if tau_low is None else tau_low
        high = [i for i in range(n) if dets[i].conf >= tau_high]
        low = [i for i in range(n) if low_floor <= dets[i].conf < tau_high]
        active = high + low
        edges = _within_group_edges(dets, high, chi2_gate, tau_euc)
        edges += _cross_group_edges(dets, low, high, chi2_gate, tau_euc)

    active_set = set(active)
    clusters: List[List[Detection]] = []
    for comp in _connected_components(n, edges):
        comp = [i for i in comp if i in active_set]  # drop detections below tau_low
        if not comp:
            continue
        comp = _sensor_unique(comp, dets)
        if len({dets[i].sensor_id for i in comp}) >= min_sensors:
            clusters.append([dets[i] for i in comp])
    return clusters
