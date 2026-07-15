"""Configuration for the greedy / RSP mutual-information planner (Corah & Michael 2021).

One flat dataclass (CleanRL style) holds every knob: the multi-robot coordinator
(sequential greedy vs RSP with n_d rounds), the per-drone candidate action set, and
the planning horizon. The single-robot planner is exhaustive enumeration of the
discretized reachable action set (exact / optimal for this small candidate set, so no
MCTS approximation is used).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InfomaxConfig:
    # --- planner ---
    method: str = "rsp"             # "greedy" (sequential, Eq 11) | "rsp" (Alg 1, n_d rounds)
    # RSP decision rounds. VERIFIED semantics (paper Fig 2): n_d = n_r -> sequential greedy
    # (one drone per round, each sees all priors); n_d = 1 -> fully parallel (no coordination).
    # More rounds -> closer to greedy. Clamped to [1, n_r] at run time.
    n_d: int = 2
    rsp_seed: int = 0              # PRNG seed for the random drone -> round partition

    # --- candidate action set per drone (continuous-drone discretization of U_i) ---
    n_directions: int = 8           # K compass directions in the reachable set
    include_stay: bool = True       # include the "stay put" action
    n_speed_rings: int = 1          # radial reach magnitudes per direction (1 = only full v_max*dt)
    v_max: float = 10.0             # per-step speed cap -> reach radius = v_max * dt

    # --- planning horizon (single-robot planner is exhaustive enumeration) ---
    # Each drone's candidate action set (single_step_offsets ** horizon trajectories) is
    # small, so the single-robot planner enumerates it exhaustively -- exact/optimal, no
    # MCTS approximation needed.
    horizon: int = 2                # l-step lookahead (paper uses 2); horizon-summed MI (Eq 28)

    # --- objective ---
    q: float = 0.05                 # CV process-noise intensity for horizon target prediction
    weight_by_phd: bool = True      # weight each target's MI by its GM-PHD component weight w
