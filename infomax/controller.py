"""Thin driver: turn GM-PHD target priors + drone poses into greedy/RSP repositioning.

For each drone it enumerates the reachable l-step trajectories, computes each
trajectory's per-step / per-target position information via an injected info_fn (the
camera measurement_information closure, so this package imports nothing from netcomm),
runs sequential_greedy / rsp over the drones, and returns each drone's executed
first-step xy. Targets are predicted forward with the CV model so the horizon MI points
sensors where targets will be. Mirrors coverage_control.CoverageController.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .config import InfomaxConfig
from .objective import cv_matrices
from .actions import drone_trajectories
from .maximizer import plan


class InfomaxController:
    def __init__(self, config: Optional[InfomaxConfig] = None):
        self.cfg = config or InfomaxConfig()

    def step(self, drones_xyz, target_means, target_covs, target_weights, area_xy, dt,
             info_fn: Callable) -> Dict:
        """One planning step.

        drones_xyz     : (N, 3) drone poses (altitude fixed).
        target_means   : (M, 4) GM-PHD state means [px,py,vx,vy].
        target_covs    : (M, 4, 4) GM-PHD state covariances (the MI priors).
        target_weights : (M,) PHD weights.
        info_fn        : (drone_xyz(3,), target_xy(2,)) -> 2x2 position information R^{-1}.
        Returns dict: new_xy (N,2), chosen (N,), gains (N,), redundancy (N,), total_mi,
        candidates (list of (n_traj, L, 2) per drone, for viz).
        """
        cfg = self.cfg
        drones = np.asarray(drones_xyz, dtype=float).reshape(-1, 3)
        N = drones.shape[0]
        means = np.asarray(target_means, dtype=float).reshape(-1, 4)
        covs = np.asarray(target_covs, dtype=float).reshape(-1, 4, 4)
        M = means.shape[0]
        L = int(cfg.horizon)
        F, Q = cv_matrices(float(dt), cfg.q)

        if N == 0 or M == 0:                               # nothing to plan against
            return {"new_xy": drones[:, :2].copy(), "chosen": np.zeros(N, int),
                    "gains": np.zeros(N), "redundancy": np.zeros(N), "total_mi": 0.0,
                    "candidates": [np.empty((0, L, 2)) for _ in range(N)]}

        # predicted target positions over the horizon (CV mean propagation)
        F_np = np.asarray(F)
        pred_pos = np.zeros((M, L, 2))
        for j in range(M):
            m = means[j].copy()
            for k in range(L):
                m = F_np @ m
                pred_pos[j, k] = m[:2]

        weights = (np.asarray(target_weights, float) if cfg.weight_by_phd else np.ones(M))

        candidate_infos, first_xy, candidates = [], [], []
        for d in range(N):
            trajs = drone_trajectories(drones[d, :2], cfg, dt, area_xy)   # (n_traj, L, 2)
            z = drones[d, 2]
            cinfo = np.zeros((len(trajs), M, L, 2, 2))
            for ti in range(len(trajs)):
                for k in range(L):
                    dpose = np.array([trajs[ti, k, 0], trajs[ti, k, 1], z])
                    for j in range(M):
                        cinfo[ti, j, k] = info_fn(dpose, pred_pos[j, k])
            candidate_infos.append(cinfo)
            first_xy.append(trajs[:, 0, :])
            candidates.append(trajs)

        res = plan(candidate_infos, covs, weights, F, Q,
                   method=cfg.method, n_d=cfg.n_d, seed=cfg.rsp_seed)
        new_xy = np.array([first_xy[d][res["chosen"][d]] for d in range(N)])
        return {"new_xy": new_xy, "chosen": res["chosen"], "gains": res["gains"],
                "redundancy": res["redundancy"], "total_mi": res["total_mi"],
                "candidates": candidates}
