"""Non-myopic minimax target tracking for state-dependent noise (Zhang & Tokekar 2016).

A standalone package (numpy; the tree is dynamic/recursive, the reusable JAX Riccati is in
evaluation/pcrlb). A single robot plans a closed-loop control POLICY by building a minimax
tree over future controls (min) and discretized measurements (max), minimizing the
worst-case trace of the Kalman posterior covariance under state-dependent measurement noise.

  riccati.py - Kalman predict/update + the state-dependent noise model (Eqs 2-7).
  sampler.py    - candidate-measurement discretization for the max levels (Fig 1).
  tree.py       - the exact minimax tree (Algorithm 1); leaf value = tr(Sigma).
  pruning.py    - alpha-beta + algebraic-redundancy pruning (Thms 2/3/5); same optimum.
  assignment.py - sequential-greedy multi-robot assignment over the per-robot tree.

Multi-robot (the authors' proposed extension, never published): greedy assignment of
robots to targets with this tree as the per-robot planner; the joint minimax is intractable.
"""
from .config import MinimaxConfig
from .riccati import (
    cv_matrices, predict, kalman_update, riccati_step, paper_noise, H,
    riccati_step_j, predict_j, kalman_update_j,
)
from .sampler import candidate_measurements, candidate_measurements_j
from .tree import action_offsets, minimax_value, minimax_value_vectorized, plan_single_target
from .pruning import minimax_value_pruned, plan_single_target_pruned
from .assignment import greedy_assignment

__all__ = [
    "MinimaxConfig",
    "cv_matrices",
    "predict",
    "kalman_update",
    "riccati_step",
    "paper_noise",
    "candidate_measurements",
    "action_offsets",
    "minimax_value",
    "minimax_value_vectorized",
    "plan_single_target",
    "minimax_value_pruned",
    "plan_single_target_pruned",
    "greedy_assignment",
    "H",
    # genuine-JAX kernels (jnp + jit) used by the vectorized live minimax
    "riccati_step_j",
    "predict_j",
    "kalman_update_j",
    "candidate_measurements_j",
]
