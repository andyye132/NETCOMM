"""Configuration for the non-myopic minimax target tracker (Zhang & Tokekar 2016).

One flat dataclass (CleanRL style). The robot plans a closed-loop control policy by
building a minimax tree over future controls (min) and discretized measurements (max),
minimizing the worst-case trace of the Kalman posterior covariance under state-dependent
measurement noise. n_meas candidate measurements per step discretize the Gaussian within
~3 sigma; eps1/eps2 trade optimality for fewer nodes (alpha and algebraic-redundancy
pruning). Multi-robot uses greedy/RSP assignment with this tree as the per-robot planner.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinimaxConfig:
    # --- planning horizon + branching ---
    horizon: int = 2                # k: control/measurement step pairs (paper uses 2)
    n_directions: int = 4           # control actions per step: K compass dirs (+ stay)
    include_stay: bool = False       # paper's U = {+e,-e in x/y} has no stay
    n_meas: int = 5                 # k candidate measurements/step (paper: 5, within 3 sigma)
    v_max: float = 10.0             # per-step reach: action displacement = v_max * dt

    # --- pruning relaxation (alpha: Thm 3; redundancy: Thm 2/5 — see pruning.py); 0 = exact ---
    eps1: float = 0.0               # alpha-pruning slack
    eps2: float = 0.0               # algebraic-redundancy slack
    use_alpha_pruning: bool = True
    use_redundancy_pruning: bool = True

    # --- objective / covariance ---
    q: float = 0.05                 # target CV process-noise intensity (Sigma_v)
    weight_by_phd: bool = True      # multi-target leaf = sum_j w_j tr(Sigma_j) (honored when
    #                                 PHD weights are passed to greedy_assignment)
