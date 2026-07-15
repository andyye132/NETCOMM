"""diffsim — a DIFFERENTIABLE core for the drone-tracking sim (pure JAX).

The main sim (netcomm.tracking) uses the GM-PHD filter, a hard field-of-view gate, and
stochastic detection sampling — all of which break gradients. This package is a small,
SEPARATE, fully-differentiable model of the same physics so you can ``jax.grad`` through a
whole episode:

  drone positions  ->  soft (sigmoid) sensing  ->  Gaussian/Kalman belief  ->  scalar loss

The loss is the total tracking uncertainty (a smooth, PCRLB-style sum of posterior position
covariance over the episode). Because every step is smooth jnp, ``jax.grad(episode_uncertainty)``
gives the direction to move the drones to track better — i.e. the sim is differentiable, and you
can optimize drone placement / trajectories by gradient descent (see ``diffsim.demo``).

It imports nothing from netcomm and does not touch the GM-PHD sim.
"""
from .config import DiffSimConfig
from .model import cv_matrices, pair_info, target_info
from .rollout import episode_uncertainty, optimize_drone_positions

__all__ = [
    "DiffSimConfig",
    "cv_matrices",
    "pair_info",
    "target_info",
    "episode_uncertainty",
    "optimize_drone_positions",
]
