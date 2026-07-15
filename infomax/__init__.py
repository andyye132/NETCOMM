"""Greedy / RSP mutual-information multi-robot planner (Corah & Michael, arXiv:2107.08550).

A standalone pure-JAX package (imports nothing from netcomm), mirroring gmphd/ and
coverage_control/. It maximizes the receding-horizon mutual-information objective over
which targets the drones observe — drones choose actions to minimize target uncertainty:

  * objective.py  - Gaussian MI = horizon-summed info-filter log-det reduction (Eq 28),
                    jitted + vmapped over targets.
  * maximizer.py  - sequential_greedy (Eq 11) and rsp(n_d) (Algorithm 1), with the
                    1/2-optimal greedy bound and the n_d-rounds RSP suboptimality. The
                    single-robot planner is exhaustive enumeration of the action set.
  * actions.py    - per-drone candidate action sets (reachable moves over the horizon).
  * controller.py - thin driver wiring tracker-driven target priors into the planner.

The netcomm glue (reading GM-PHD tracks, injecting the camera measurement_information)
lives in netcomm/tracking/repositioner.py, exactly as gmphd is wrapped by tracker.py.
"""
from .config import InfomaxConfig
from .objective import cv_matrices, target_horizon_mi, set_objective
from .maximizer import sequential_greedy, rsp, plan
from .actions import single_step_offsets, drone_trajectories
from .controller import InfomaxController

__all__ = [
    "InfomaxConfig",
    "InfomaxController",
    "cv_matrices",
    "target_horizon_mi",
    "set_objective",
    "sequential_greedy",
    "rsp",
    "plan",
    "single_step_offsets",
    "drone_trajectories",
]
