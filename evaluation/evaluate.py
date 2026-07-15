"""Top-level ground-truth evaluation of a recorded tracking run.

Consumes a ``result`` dict from netcomm.tracking.run_* (which already records, per
frame, the TRUE target positions, the drone poses, and the GM-PHD estimates) plus
the sensor config, and computes — entirely offline, against ground truth — the three
complementary scores:

  1. Obtainable information (repositioner quality): the recursive PCRLB along the true
     tracks with the realized drone geometry. Dynamic (velocity/accel via F, Q).
  2. Accuracy (tracker quality): GOSPA (alpha=2) per frame + OSPA^2 over windows.
  3. Efficiency: achieved localization RMSE vs the PCRLB position bound.

Because the sim knows the truth, every number here is a ground-truth measure of how
good a given (tracker, repositioner) pair actually is.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np

from scipy.optimize import linear_sum_assignment

from .config import EvalConfig
from .gospa import gospa
from .pcrlb import pcrlb_all_targets
from .ospa2 import stitch_estimate_tracks, ospa2


def _true_tracks(frames, area_xy=None) -> List[Dict[int, np.ndarray]]:
    """Per-IDENTITY true tracks, each a {frame_index: position(2,)} dict.

    Preferred path: frames carry ``target_ids`` (recorded by the run_* runners).
    A respawn teleport allocates a fresh id in paths.py, so the replacement object
    becomes a NEW track here instead of silently continuing the old row — which
    would otherwise corrupt every identity-based metric (PCRLB info accumulation,
    OSPA^2, MOTA/IDF1, HOTA) across the position jump.

    Legacy fallback (no target_ids): the stable row-order contract, with a
    teleport heuristic that WARNS if a row jumps far enough that identity churn
    is likely (metrics from such a recording are suspect)."""
    if "target_ids" in frames[0]:
        tracks: Dict[int, Dict[int, np.ndarray]] = {}
        for k, fr in enumerate(frames):
            pos = np.asarray(fr["targets"], dtype=float).reshape(-1, 2)
            ids = np.asarray(fr["target_ids"]).reshape(-1)
            if ids.shape[0] != pos.shape[0]:
                raise ValueError(f"frame {k}: {ids.shape[0]} target_ids for "
                                 f"{pos.shape[0]} targets")
            for j in range(pos.shape[0]):
                tracks.setdefault(int(ids[j]), {})[k] = pos[j]
        return [tracks[i] for i in sorted(tracks)]

    counts = {int(np.asarray(fr["targets"]).reshape(-1, 2).shape[0]) for fr in frames}
    if len(counts) != 1:
        raise ValueError("evaluate() needs a constant target count across frames "
                         f"(got {sorted(counts)}) when frames lack target_ids.")
    n = counts.pop()
    rows = [np.array([np.asarray(fr["targets"]).reshape(-1, 2)[j] for fr in frames])
            for j in range(n)]
    if len(frames) > 1 and n > 0:
        if area_xy is not None:
            xmn, xmx, ymn, ymx = (float(a) for a in area_xy)
        else:                                    # fall back to the observed extent
            allp = np.concatenate(rows, axis=0)
            (xmn, ymn), (xmx, ymx) = allp.min(0), allp.max(0)
        diag = float(np.hypot(xmx - xmn, ymx - ymn))
        worst = max(float(np.linalg.norm(np.diff(r, axis=0), axis=1).max()) for r in rows)
        if worst > 0.25 * max(diag, 1e-9):
            warnings.warn(
                "true-target row jumped {:.1f} m in one frame (likely a respawn "
                "teleport); this recording lacks target_ids, so identity-based "
                "metrics (PCRLB/OSPA^2/MOT/HOTA) may be corrupted. Re-record with "
                "the current runners.".format(worst), RuntimeWarning, stacklevel=3)
    return [{k: r[k] for k in range(len(frames))} for r in rows]


def _estimate_positions(frame) -> np.ndarray:
    return np.array([np.asarray(p, dtype=float) for (p, _P, _w) in frame["estimates"]]) \
        if frame["estimates"] else np.zeros((0, 2))


def evaluate(result: Dict, sensor_cfg=None, cfg: Optional[EvalConfig] = None,
             dt: Optional[float] = None) -> Dict:
    """Evaluate a recorded run. Returns per-frame series + summary scalars.

    sensor_cfg / dt default to the values RECORDED in ``result`` by the run_*
    runners, so the bound is guaranteed to use the same sensor model and step
    size the run was generated with. Passing them explicitly overrides the
    recording (e.g. for what-if bounds); a run recorded before these fields
    existed must supply both explicitly.

    The PCRLB's q is likewise synced to the run: with cfg=None the recorded
    GM-PHD q is adopted (efficiency stays apples-to-apples per EvalConfig's
    lockstep rule); an explicitly-passed cfg whose q differs from the recorded
    filter q triggers a bias warning rather than silently skewing efficiency."""
    frames = result["frames"]
    T = len(frames)
    if T == 0:
        raise ValueError("no frames to evaluate")
    if sensor_cfg is None:
        sensor_cfg = result.get("sensor_cfg")
        if sensor_cfg is None:
            raise ValueError("result carries no sensor_cfg; pass sensor_cfg explicitly")
    if dt is None:
        dt = result.get("dt")
        if dt is None:
            raise ValueError("result carries no dt; pass dt explicitly")
    dt = float(dt)
    filter_q = getattr(result.get("gmphd_cfg"), "q", None)
    if cfg is None:
        cfg = EvalConfig(q=float(filter_q)) if filter_q is not None else EvalConfig()
    elif filter_q is not None and abs(cfg.q - float(filter_q)) > 1e-12:
        warnings.warn(
            f"EvalConfig.q={cfg.q} != the run's GMPHDConfig.q={filter_q}: the PCRLB bound "
            "assumes different target dynamics than the filter, so efficiency "
            "(achieved/bound) is biased (see EvalConfig's lockstep note).",
            RuntimeWarning, stacklevel=2)

    true_tracks = _true_tracks(frames, area_xy=result.get("area_xy"))
    n_t = len(true_tracks)
    alive = np.zeros((n_t, T), dtype=bool)         # (track, frame) existence mask
    for j, tr in enumerate(true_tracks):
        for k in tr:
            alive[j, k] = True
    drones_per_frame = [fr["drones"] for fr in frames]

    # --- 2a. GOSPA (alpha=2) per frame: localization / missed / false ---
    g = [gospa(_estimate_positions(fr), np.asarray(fr["targets"]).reshape(-1, 2),
               cfg.gospa_c, cfg.gospa_p, cfg.gospa_alpha) for fr in frames]
    gospa_total = np.array([x["total"] for x in g])
    gospa_loc = np.array([x["localization"] for x in g])
    missed = np.array([x["missed"] for x in g])
    false = np.array([x["false"] for x in g])
    n_matched = np.array([x["n_matched"] for x in g])

    # --- 1. PCRLB information bound along the true tracks ---
    # segment-aware: each identity track runs its own recursion over its lifespan
    # (a respawned target restarts from the diffuse prior — no information carries
    # across the teleport), padded with NaN/0 outside the lifespan.
    bound = pcrlb_all_targets(true_tracks, drones_per_frame, sensor_cfg, cfg, dt=dt)
    bound_rmse = bound["pos_rmse_bound_mean"]                     # per-frame mean over LIVE targets
    info_gain = bound["info_gain_total"]
    bound_mat = (np.array([pt["pos_rmse_bound"] for pt in bound["per_target"]])
                 if bound["per_target"] else np.zeros((0, T)))    # (n_tracks, T), NaN when dead

    # --- 3. achieved error & efficiency, ALIGNED per (track, frame) ---
    # assign each frame's estimates to the true tracks ALIVE that frame (within cutoff)
    # so achieved error and the bound are compared at the SAME track-frames.
    achieved = np.full((n_t, T), np.nan)
    for k, fr in enumerate(frames):
        E = _estimate_positions(fr)
        idx = [j for j in range(n_t) if alive[j, k]]
        if E.shape[0] == 0 or not idx:
            continue
        P = np.array([true_tracks[j][k] for j in idx])
        D = np.linalg.norm(E[:, None, :] - P[None, :, :], axis=2)        # (m, n_alive)
        rr, cc = linear_sum_assignment(np.where(D <= cfg.gospa_c, D, 1e9))
        for r, c in zip(rr, cc):
            if D[r, c] <= cfg.gospa_c:
                achieved[idx[c], k] = D[r, c]
    matched = ~np.isnan(achieved)
    tracked_fraction = float(matched[alive].mean()) if alive.any() else 0.0
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        # frames where nothing matched produce benign all-NaN slices
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        # per-frame RMS over the targets observed that frame; achieved (2D Euclidean error)
        # and bound (2D radial RMS) are the SAME convention, so efficiency ~ 1 at the bound.
        achieved_rmse = np.sqrt(np.nanmean(np.where(matched, achieved ** 2, np.nan), axis=0)) \
            if n_t else np.full(T, np.nan)
        bound_obs_per_frame = np.sqrt(np.nanmean(np.where(matched, bound_mat ** 2, np.nan), axis=0)) \
            if n_t else np.full(T, np.nan)
        efficiency = achieved_rmse / bound_obs_per_frame
    if matched.any():
        achieved_rms = float(np.sqrt(np.mean(achieved[matched] ** 2)))
        bound_rms_obs = float(np.sqrt(np.mean(bound_mat[matched] ** 2)))
        efficiency_mean = achieved_rms / bound_rms_obs
        # RMS (squared) aggregation is dominated by rare single-frame GM-PHD extraction
        # transients (e.g. a momentary under-extraction on two converging targets), so also
        # report the median of the per-frame efficiency ratio: a robust "typical" achieved/bound
        # that reflects sustained tracker quality rather than a one-frame outlier.
        with np.errstate(invalid="ignore"):
            efficiency_median = (float(np.nanmedian(efficiency))
                                 if np.isfinite(efficiency).any() else float("nan"))
    else:
        achieved_rms = bound_rms_obs = efficiency_mean = efficiency_median = float("nan")

    # --- 2b. OSPA^2 (track-level) ---
    est_tracks = stitch_estimate_tracks(frames, cfg.stitch_gate)
    true_track_dicts = true_tracks                # already {frame: pos} identity segments
    ospa2_series = ospa2(est_tracks, true_track_dicts, T, cfg.gospa_c, cfg.gospa_p,
                         cfg.ospa2_window)

    # --- 2c. CV-MOT translation metrics (MOTA/MOTP/IDF1 + HOTA) on the SAME stitched tracks ---
    mot = None
    hota_res = None
    if cfg.compute_mot:
        from .hota import hota as _hota
        hota_res = _hota(est_tracks, true_track_dicts, T, tau=cfg.mot_tau)   # self-contained, no extra dep
        try:
            from .mot import clear_mot_id_metrics
            mot = clear_mot_id_metrics(est_tracks, true_track_dicts, T, tau=cfg.mot_tau)
        except ImportError:
            mot = None                            # py-motmetrics not installed -> skip gracefully

    def _nanmean(a):
        a = np.asarray(a, dtype=float)
        return float(np.nanmean(a)) if a.size and not np.all(np.isnan(a)) else float("nan")

    return {
        "n_frames": T,
        "n_targets": len(true_tracks),      # identity segments (respawn = new track)
        # per-frame series
        "gospa": gospa_total, "gospa_localization": gospa_loc,
        "missed": missed, "false": false, "n_matched": n_matched,
        "achieved_rmse": achieved_rmse,
        "bound_rmse": bound_rmse, "info_gain": info_gain,
        "efficiency": efficiency, "ospa2": ospa2_series,
        "tracked_fraction": tracked_fraction,
        # summary scalars
        "summary": {
            "gospa_mean": float(np.mean(gospa_total)),
            "gospa_localization_mean": float(np.mean(gospa_loc)),
            "missed_total": int(missed.sum()), "false_total": int(false.sum()),
            "tracked_fraction": tracked_fraction,
            "achieved_rmse_mean": achieved_rms,                    # 2D RMS error where tracked
            "bound_rmse_mean": _nanmean(bound_rmse),               # live target-frames (repositioner score)
            "bound_rmse_observed": bound_rms_obs,                  # 2D RMS bound where tracked (vs achieved)
            "efficiency_mean": efficiency_mean,                    # achieved_rms / bound_rms (~1 optimal)
            "efficiency_median": efficiency_median,                # robust per-frame achieved/bound (~1 optimal)
            "info_gain_per_step": float(np.mean(info_gain)),
            "info_gain_cumulative": float(bound.get("info_gain_cumulative", np.sum(info_gain))),
            "ospa2_mean": float(np.mean(ospa2_series)),
            "n_est_tracks": len(est_tracks),
            # CV-MOT translation metrics (None if py-motmetrics is not installed)
            "mota": (mot or {}).get("mota"), "motp": (mot or {}).get("motp"),
            "idf1": (mot or {}).get("idf1"), "id_switches": (mot or {}).get("id_switches"),
            "hota": (hota_res or {}).get("hota"), "deta": (hota_res or {}).get("deta"),
            "assa": (hota_res or {}).get("assa"),
        },
        "mot": mot,
        "hota": hota_res,
        "_bound": bound,
    }
