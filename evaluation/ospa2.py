"""OSPA^2 — the track-level (dynamic) multi-target error metric (Beard, Vo & Vo 2017).

OSPA at a single frame is a snapshot. OSPA^2 ("OSPA-on-OSPA") computes, over a
sliding time window, an OSPA distance between sets of whole TRACKS, where the base
distance between two tracks is their time-averaged (cutoff) position distance over
the window. This captures track switching, fragmentation, and persistence that a
per-frame metric cannot — the dynamic counterpart to GOSPA.

Caveat: the GM-PHD filter is label-free, so its per-frame estimates carry no track
identity. We stitch them into tracks here with a simple greedy nearest-neighbour
associator (an evaluation-time heuristic). True tracks come from the stable target
row order. A labelled tracker (GLMB/LMB) would remove the need for stitching.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def stitch_estimate_tracks(frames: Sequence, gate: float = 8.0,
                           max_gap: int = 3) -> List[Dict[int, np.ndarray]]:
    """Greedy nearest-neighbour stitch of per-frame GM-PHD estimates into tracks.

    Returns a list of tracks, each a dict {frame_index: position(2,)}.
    """
    tracks: List[Dict] = []                  # each: {"pts": {k: pos}, "last_pos", "last_frame"}
    for k, fr in enumerate(frames):
        ests = [np.asarray(p, dtype=float) for (p, _P, _w) in fr["estimates"]]
        open_idx = [i for i, t in enumerate(tracks) if k - t["last_frame"] <= max_gap]
        pairs = []
        for oi in open_idx:
            last = tracks[oi]["last_pos"]
            for ei, e in enumerate(ests):
                d = float(np.linalg.norm(last - e))
                if d <= gate:
                    pairs.append((d, oi, ei))
        pairs.sort()
        used_tracks, used_est = set(), set()
        for _d, oi, ei in pairs:
            if oi in used_tracks or ei in used_est:
                continue
            tracks[oi]["pts"][k] = ests[ei]
            tracks[oi]["last_pos"] = ests[ei]
            tracks[oi]["last_frame"] = k
            used_tracks.add(oi)
            used_est.add(ei)
        for ei, e in enumerate(ests):
            if ei not in used_est:
                tracks.append({"pts": {k: e}, "last_pos": e, "last_frame": k})
    return [t["pts"] for t in tracks]


def _track_base_distance(a: Dict, b: Dict, window: Sequence[int], c: float, p: float) -> float:
    """Time-averaged cutoff distance between two tracks over the window (existence-aware)."""
    num, cnt = 0.0, 0
    for t in window:
        pa, pb = a.get(t), b.get(t)
        if pa is None and pb is None:
            continue
        d = c if (pa is None or pb is None) else min(c, float(np.linalg.norm(pa - pb)))
        num += d ** p
        cnt += 1
    return 0.0 if cnt == 0 else (num / cnt) ** (1.0 / p)


def _present(track: Dict, window: Sequence[int]) -> bool:
    return any(t in track for t in window)


def _ospa_sets(A: List[Dict], B: List[Dict], window, c: float, p: float) -> float:
    """OSPA distance between two track sets using the time-averaged base distance."""
    m, n = len(A), len(B)
    if m == 0 and n == 0:
        return 0.0
    if m > n:                                  # ensure m <= n (OSPA normalizes by max)
        A, B, m, n = B, A, n, m
    if m == 0:
        return c                               # all-cardinality mismatch
    D = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = min(c, _track_base_distance(A[i], B[j], window, c, p)) ** p
    rows, cols = linear_sum_assignment(D)
    loc = D[rows, cols].sum()
    card = (c ** p) * (n - m)
    return ((loc + card) / n) ** (1.0 / p)


def ospa2(est_tracks: List[Dict], true_tracks: List[Dict], n_frames: int,
          c: float = 10.0, p: float = 2.0, window: int = 10) -> np.ndarray:
    """OSPA^2 time series: one value per sliding window start (length n_frames)."""
    series = np.zeros(n_frames)
    for start in range(n_frames):
        w = list(range(start, min(start + window, n_frames)))
        A = [t for t in est_tracks if _present(t, w)]
        B = [t for t in true_tracks if _present(t, w)]
        series[start] = _ospa_sets(A, B, w, c, p)
    return series
