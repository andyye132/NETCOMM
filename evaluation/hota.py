"""HOTA — Higher Order Tracking Accuracy (Luiten et al., IJCV 2021).

The modern MOTChallenge/KITTI headline metric. It balances DETECTION and ASSOCIATION
(MOTA over-weights detection; IDF1 over-weights association):

    HOTA(alpha) = sqrt( DetA(alpha) * AssA(alpha) )

averaged over 19 localization thresholds alpha in {0.05, 0.10, ..., 0.95}. Range [0, 1],
higher is better. DetA is detection accuracy TP/(TP+FN+FP); AssA is the TP-weighted mean
of the per-pair association Jaccard; LocA is the mean match similarity (reported, not in
the product).

Self-contained numpy/scipy port of TrackEval's trackeval/metrics/hota.py, adapted for 2D
POINT tracks: the per-frame similarity is S = clip(1 - d/tau, 0, 1) (the standard
distance->similarity mapping; tau = GOSPA's cutoff). TrackEval is built for bounding-box
video benchmarks, so a faithful point-data port (validated against TrackEval's exact
arithmetic in the tests) is cleaner than vendoring it. Like MOTA/IDF1, HOTA needs tracks
WITH IDS, so it consumes the stitched GM-PHD tracks — a CV-family "translation" metric.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

_ALPHAS = np.arange(0.05, 0.99, 0.05)      # 19 thresholds 0.05 .. 0.95 (TrackEval grid)
_EPS = float(np.finfo(float).eps)


def hota(est_tracks: List[Dict], true_tracks: List[Dict], n_frames: int,
         tau: float = 10.0) -> Dict:
    """HOTA (+ DetA, AssA, LocA) from stitched estimate tracks vs true tracks.

    est_tracks / true_tracks : list of {frame_index: position(2,)} (track id = list index).
    tau : localization scale (m); per-frame similarity S = max(0, 1 - dist/tau).
    Returns scalar means (hota/deta/assa/loca), HOTA(0) [alpha=0.05], and per-alpha arrays.
    """
    nG, nH = len(true_tracks), len(est_tracks)
    A = len(_ALPHAS)
    if nG == 0 and nH == 0:
        return {"hota": 1.0, "deta": 1.0, "assa": 1.0, "loca": 1.0,
                "hota_alpha0": 1.0, "alphas": _ALPHAS}

    # per frame: present gt ids, present hyp ids, and the [0,1] similarity matrix
    per_frame = []
    for f in range(n_frames):
        g_ids = [i for i, tr in enumerate(true_tracks) if f in tr]
        h_ids = [j for j, tr in enumerate(est_tracks) if f in tr]
        if g_ids and h_ids:
            G = np.array([true_tracks[i][f] for i in g_ids])
            Hh = np.array([est_tracks[j][f] for j in h_ids])
            d = np.linalg.norm(G[:, None, :] - Hh[None, :, :], axis=2)
            S = np.clip(1.0 - d / tau, 0.0, 1.0)
        else:
            S = np.zeros((len(g_ids), len(h_ids)))
        per_frame.append((np.array(g_ids, int), np.array(h_ids, int), S))

    # Pass A — global alignment score per (gt-id, hyp-id): how consistently they co-occur
    pmc = np.zeros((nG, nH))
    gc = np.zeros(nG)
    tc = np.zeros(nH)
    for g_ids, h_ids, S in per_frame:
        if len(g_ids) and len(h_ids):
            denom = S.sum(0)[None, :] + S.sum(1)[:, None] - S
            sio = np.where(denom > _EPS, S / np.where(denom > _EPS, denom, 1.0), 0.0)
            pmc[g_ids[:, None], h_ids[None, :]] += sio
        gc[g_ids] += 1
        tc[h_ids] += 1
    gas = pmc / (gc[:, None] + tc[None, :] - pmc + _EPS)

    # Pass B — per-frame Hungarian on the alignment-weighted score, then threshold per alpha
    TP = np.zeros(A)
    FN = np.zeros(A)
    FP = np.zeros(A)
    LocA = np.zeros(A)
    mcounts = [np.zeros((nG, nH)) for _ in range(A)]
    for g_ids, h_ids, S in per_frame:
        if len(g_ids) and len(h_ids):
            mr, mc = linear_sum_assignment(-(gas[g_ids[:, None], h_ids[None, :]] * S))
        else:
            mr = mc = np.array([], int)
        for a, al in enumerate(_ALPHAS):
            mask = (S[mr, mc] >= al - _EPS) if len(mr) else np.array([], bool)
            amr, amc = mr[mask], mc[mask]
            n = int(len(amr))
            TP[a] += n
            FN[a] += len(g_ids) - n
            FP[a] += len(h_ids) - n
            if n:
                LocA[a] += S[amr, amc].sum()
                mcounts[a][g_ids[amr], h_ids[amc]] += 1

    DetA = TP / np.maximum(1, TP + FN + FP)
    AssA = np.zeros(A)
    for a in range(A):
        m = mcounts[a]
        cell = m / np.maximum(1, gc[:, None] + tc[None, :] - m)        # per-pair association Jaccard
        AssA[a] = np.sum(m * cell) / max(1, TP[a])
    LocA = np.maximum(1e-10, LocA) / np.maximum(1e-10, TP)
    HOTA = np.sqrt(DetA * AssA)
    return {"hota": float(HOTA.mean()), "deta": float(DetA.mean()),
            "assa": float(AssA.mean()), "loca": float(LocA.mean()),
            "hota_alpha0": float(HOTA[0]), "alphas": _ALPHAS,
            "hota_per_alpha": HOTA, "deta_per_alpha": DetA, "assa_per_alpha": AssA}
