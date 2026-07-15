"""Standalone Voronoi-based coverage control (Cortes et al. 2004) in pure JAX.

The "isotropic Voronoi based coverage" repositioning method: drive a team of
mobile sensors to a centroidal Voronoi configuration that minimizes the coverage
cost  H = sum_i int_{V_i} ||q - p_i||^2 phi(q) dq  over an importance density phi,
by moving each sensor toward the phi-weighted centroid of its Voronoi cell.

Self-contained (imports nothing from netcomm or the tracker). The covariance/
estimate -> phi mapping and the drone bookkeeping live in the netcomm.tracking
adapter; this package is just the math + a thin driver, mirroring how gmphd/ is a
standalone filter wrapped by netcomm.tracking.tracker.
"""
from .config import CoverageConfig
from .controller import CoverageController
from .voronoi import (
    make_grid,
    build_phi_grid,
    voronoi_assign,
    partition_centroids_cost,
    coverage_cost,
    lloyd_step,
    control_step,
)

__all__ = [
    "CoverageConfig",
    "CoverageController",
    "make_grid",
    "build_phi_grid",
    "voronoi_assign",
    "partition_centroids_cost",
    "coverage_cost",
    "lloyd_step",
    "control_step",
]
