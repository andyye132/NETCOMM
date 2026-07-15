"""Computer-vision MOT metrics (MOTA, MOTP, IDF1) via py-motmetrics.

These are the CLEAR-MOT (Bernardin & Stiefelhagen 2008) and ID (Ristani et al. 2016)
metrics from the vision tracking-by-detection community. Unlike GOSPA/OSPA (which are
label-free set distances), they need tracks WITH PERSISTENT IDS — so they consume the
stitched GM-PHD tracks and the true tracks, not the raw per-frame estimates. They are
"translation" metrics for cross-comparability with the vision/aerial-MOT literature;
the RFS metrics (GOSPA, OSPA^2) + PCRLB remain the headline for this RFS/sensor-mgmt sim.

We WRAP the canonical py-motmetrics implementation rather than reimplement, so the
numbers match the published definitions exactly (the sticky-match + Hungarian IDSW
policy and the global IDF1 assignment have subtle edge cases). MOTP is reported as
mean Euclidean distance (lower = better), in the same units as GOSPA's cutoff.

Note (HOTA): the modern MOTChallenge/KITTI headline metric is HOTA, which needs the
heavier TrackEval package and is not included here.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def _require_motmetrics():
    try:
        import motmetrics as mm
    except ImportError as e:                      # pragma: no cover
        raise ImportError("MOTA/MOTP/IDF1 need py-motmetrics: `uv pip install motmetrics`") from e
    return mm


def clear_mot_id_metrics(est_tracks: List[Dict], true_tracks: List[Dict], n_frames: int,
                         tau: float = 10.0) -> Dict:
    """MOTA, MOTP, IDF1 (+ IDP, IDR, counts) from stitched estimate tracks vs true tracks.

    est_tracks / true_tracks : list of {frame_index: position(2,)} (track id = list index).
    tau : gating distance (m); estimate-truth pairs farther than tau are not matchable.
    MOTP is the mean Euclidean match distance (lower better), comparable to GOSPA's cutoff.
    """
    mm = _require_motmetrics()
    acc = mm.MOTAccumulator(auto_id=True)
    for t in range(n_frames):
        gt_ids = [j for j, tr in enumerate(true_tracks) if t in tr]
        hy_ids = [i for i, tr in enumerate(est_tracks) if t in tr]
        gt_pts = np.array([true_tracks[j][t] for j in gt_ids]) if gt_ids else np.zeros((0, 2))
        hy_pts = np.array([est_tracks[i][t] for i in hy_ids]) if hy_ids else np.zeros((0, 2))
        if len(gt_pts) and len(hy_pts):
            D = np.linalg.norm(gt_pts[:, None, :] - hy_pts[None, :, :], axis=2)  # linear Euclidean
            D[D > tau] = np.nan                   # gate: non-matchable above the cutoff
        else:
            D = np.zeros((len(gt_pts), len(hy_pts)))
        acc.update(gt_ids, hy_ids, D)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc, name="seq",
        metrics=["mota", "motp", "idf1", "idp", "idr",
                 "num_misses", "num_false_positives", "num_switches", "num_matches"])
    row = summary.loc["seq"]

    def _f(x):
        v = float(x)
        return v if np.isfinite(v) else float("nan")

    return {
        "mota": _f(row["mota"]), "motp": _f(row["motp"]), "idf1": _f(row["idf1"]),
        "idp": _f(row["idp"]), "idr": _f(row["idr"]),
        "misses": int(row["num_misses"]), "false_positives": int(row["num_false_positives"]),
        "id_switches": int(row["num_switches"]), "matches": int(row["num_matches"]),
    }
