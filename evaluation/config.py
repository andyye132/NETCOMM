"""Configuration for the ground-truth tracking-evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    # --- PCRLB / information bound ---
    # NOTE: q and the priors set the bound's assumed target dynamics. For efficiency
    # (achieved RMSE / bound) to read as "how close to optimal the filter is", q/priors
    # should reflect the TRUE target process noise (and roughly match the filter's CV/q);
    # a tighter bound q than the filter assumes makes efficiency look artificially high.
    # The true sim motion is deterministic (CV + reflection / parametric paths, zero
    # process noise), so q is a shared CONVENTION: it must equal GMPHDConfig.q (0.01)
    # or the efficiency ratio is biased. Keep the two in lockstep.
    motion: str = "cv"              # target-state model for the bound: "cv" [pos,vel] | "ca" [pos,vel,acc]
    q: float = 0.01                 # process-noise intensity for the bound's Q (must be > 0);
    #                                 = GMPHDConfig.q so achieved/bound is apples-to-apples
    prior_pos_var: float = 25.0     # diffuse prior variance (m^2) on position
    prior_vel_var: float = 100.0    # diffuse prior variance on velocity
    prior_acc_var: float = 100.0    # diffuse prior variance on acceleration (CA only)

    # --- GOSPA / OSPA error metrics ---
    gospa_c: float = 10.0           # cutoff distance (m)
    gospa_p: float = 2.0            # order
    gospa_alpha: float = 2.0        # 2.0 -> localization/missed/false decomposition

    # --- OSPA^2 (track-level) ---
    ospa2_window: int = 10          # sliding-window length (frames)
    stitch_gate: float = 8.0        # nearest-neighbour gate (m) for stitching estimates into tracks

    # --- CV-MOT metrics (MOTA/MOTP/IDF1 via py-motmetrics; need stitched tracks) ---
    compute_mot: bool = True        # compute the vision MOT metrics (skipped if motmetrics absent)
    mot_tau: float = 10.0           # gating distance (m) for MOT matching (analogous to gospa_c)
