"""Posterior Cramer-Rao Lower Bound (PCRLB) — the dynamic, ground-truth information bound.

Tichavsky, Muravchik & Nehorai (1998). For a linear-Gaussian target (CV/CA motion,
position measurements) the recursion collapses to the information-filter form

    J_k = ( F J_{k-1}^{-1} F^T + Q )^{-1}  +  sum_drones  H^T M_{d,k} H

where M_{d,k} is drone d's expected position information about the target at step k
(geometry- and LoS/NLoS-aware, from netcomm.tracking.sensors.measurement_information).
Missed detections are folded in by the p_D-weighted expected information — the standard
information-reduction-factor (IRF) approximation (Hernandez/Farina/Ristic): an
optimistic (tight) bound when p_D < 1; clutter is not modeled (use clutter_rate = 0).
J_k^{-1} is the best achievable estimation-error covariance: the information the
drones' placement made obtainable, accounting for target motion. Velocity (and, with
the CA model, acceleration) enter through F, Q in the prediction term, so the bound
is inherently dynamic, not a snapshot: a fast/maneuvering target (large Q) erodes
information between looks.

Evaluated along the TRUE trajectory with the REALIZED drone geometry, this is a
ground-truth quantity the simulator can compute because it knows the true states.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .motion import model_matrices


def _slogdet(A: np.ndarray) -> float:
    return float(np.linalg.slogdet(A)[1])


def pcrlb_track(true_positions: Sequence, drones_per_frame: Sequence,
                sensor_cfg, motion: str = "cv", q: float = 0.05, dt: float = 1.0,
                prior_pos_var: float = 25.0, prior_vel_var: float = 100.0,
                prior_acc_var: float = 100.0) -> Dict:
    """Recursive posterior FIM for one target along its true track.

    true_positions   : (T, 2) the target's true ground positions.
    drones_per_frame : length-T sequence of (N, 3) drone poses at each frame.
    sensor_cfg       : CameraSensorConfig (supplies the geometry/LoS measurement info).

    Returns per-frame arrays: info_logdet (T,), pos_rmse_bound (T,) = sqrt(var_x + var_y)
    [2D radial RMS position-error bound], info_gain (T,) = 1/2 log det(J_k J_pred^{-1}) [Gaussian mutual
    information from this step's looks], n_observing (T,) drones in view.
    """
    from netcomm.tracking.sensors import measurement_information

    if q <= 0:
        raise ValueError("pcrlb_track needs q > 0 (process noise) for a non-singular bound")
    pos = np.asarray(true_positions, dtype=float).reshape(-1, 2)
    T = pos.shape[0]
    F, Q, H = model_matrices(motion, dt, q)
    s = F.shape[0]

    diag = [1.0 / prior_pos_var] * 2 + [1.0 / prior_vel_var] * 2
    if motion == "ca":
        diag += [1.0 / prior_acc_var] * 2
    J = np.diag(diag)

    logdet = np.zeros(T)
    rmse = np.zeros(T)
    gain = np.zeros(T)
    n_obs = np.zeros(T, dtype=int)

    for k in range(T):
        J_pred = np.linalg.inv(F @ np.linalg.inv(J) @ F.T + Q)
        M = np.zeros((2, 2))                       # summed expected position information
        cnt = 0
        for d in np.asarray(drones_per_frame[k], dtype=float).reshape(-1, 3):
            Md = measurement_information(d, pos[k], sensor_cfg)
            if np.any(Md):
                M += Md
                cnt += 1
        J = J_pred + H.T @ M @ H                   # embed position info into the full state
        cov = np.linalg.inv(J)
        logdet[k] = _slogdet(J)
        # 2D radial RMS position bound sqrt(var_x + var_y) = sqrt(E||p_hat - p||^2),
        # the same convention as the empirical Euclidean error it is compared against.
        rmse[k] = float(np.sqrt(np.trace(cov[:2, :2])))
        gain[k] = 0.5 * (_slogdet(J) - _slogdet(J_pred))
        n_obs[k] = cnt

    return {"info_logdet": logdet, "pos_rmse_bound": rmse,
            "info_gain": gain, "n_observing": n_obs}


def pcrlb_all_targets(true_tracks: Sequence, drones_per_frame: Sequence, sensor_cfg,
                      cfg, dt: float = 1.0) -> Dict:
    """Run the PCRLB recursion for every target track and aggregate.

    true_tracks : one entry per identity track, either
      - an (T, 2) array covering every frame (legacy, stable-row targets), or
      - a {frame_index: position(2,)} dict covering a CONTIGUOUS lifespan (identity
        segment: a respawned target is a new track). The recursion runs only over
        the lifespan, restarting from the diffuse prior — no Fisher information
        carries across a respawn teleport. Outputs are padded to the full episode
        (NaN bound / 0 gain when the track does not exist).

    Returns per-target results plus aggregate per-frame series (summed information
    gain over live tracks, mean position bound RMSE across live tracks).
    """
    T = len(drones_per_frame)
    per_target: List[Dict] = []
    for tr in true_tracks:
        if isinstance(tr, dict):
            ks = sorted(tr)
            t0, t1 = ks[0], ks[-1]
            if ks != list(range(t0, t1 + 1)):
                raise ValueError("true track has gaps; identity segments must be contiguous")
            seg_pos = np.array([tr[k] for k in ks], dtype=float)
            seg = pcrlb_track(seg_pos, drones_per_frame[t0:t1 + 1], sensor_cfg,
                              motion=cfg.motion, q=cfg.q, dt=dt,
                              prior_pos_var=cfg.prior_pos_var,
                              prior_vel_var=cfg.prior_vel_var,
                              prior_acc_var=cfg.prior_acc_var)
            full = {"info_logdet": np.full(T, np.nan),
                    "pos_rmse_bound": np.full(T, np.nan),
                    "info_gain": np.zeros(T),
                    "n_observing": np.zeros(T, dtype=int)}
            for key in full:
                full[key][t0:t1 + 1] = seg[key]
            per_target.append(full)
        else:
            per_target.append(
                pcrlb_track(tr, drones_per_frame, sensor_cfg, motion=cfg.motion, q=cfg.q,
                            dt=dt, prior_pos_var=cfg.prior_pos_var,
                            prior_vel_var=cfg.prior_vel_var,
                            prior_acc_var=cfg.prior_acc_var))
    if not per_target:
        return {"per_target": [], "info_gain_total": np.zeros(0),
                "pos_rmse_bound_mean": np.zeros(0)}
    info_gain_total = np.sum([r["info_gain"] for r in per_target], axis=0)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        pos_rmse_bound_mean = np.nanmean([r["pos_rmse_bound"] for r in per_target], axis=0)
    return {"per_target": per_target, "info_gain_total": info_gain_total,
            "pos_rmse_bound_mean": pos_rmse_bound_mean,
            "info_gain_cumulative": float(np.sum(info_gain_total))}
