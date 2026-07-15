"""Thin stateful driver around the pure Voronoi/Lloyd math.

Caches the quadrature grid for a given area and runs one coverage-control step
per call: rasterize phi from the target estimates, partition into Voronoi cells,
and move each sensor toward its cell centroid (speed-capped descent, or a Lloyd
jump). Pure: takes sensor xy + estimate centers/weights, returns numpy arrays.
The netcomm glue (reading the tracker belief, writing back into the drones) lives
in netcomm/tracking/repositioner.py, exactly as gmphd is wrapped by tracker.py.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import jax.numpy as jnp

from .config import CoverageConfig
from .voronoi import make_grid, build_phi_grid, partition_centroids_cost, capped_descent


class CoverageController:
    def __init__(self, config: Optional[CoverageConfig] = None):
        self.cfg = config or CoverageConfig()
        self._grid_pts = None
        self._cell_area = None
        self._nx = self._ny = int(self.cfg.grid_res)
        self._area = None

    def _ensure_grid(self, area_xy):
        area = tuple(float(a) for a in area_xy)
        if self._grid_pts is None or self._area != area:
            self._grid_pts, self._cell_area = make_grid(area, self._nx, self._ny)
            self._area = area
        return self._grid_pts, self._cell_area

    def step(self, positions_xy, centers, weights, area_xy, dt):
        """One coverage step.

        positions_xy : (N, 2) current sensor ground positions.
        centers      : (M, 2) target-estimate means seeding phi (may be empty).
        weights      : (M,) PHD weights for the centers.
        Returns dict with new_xy (N, 2), labels (ny, nx) int, centroids (N, 2),
        cost (float), phi (ny, nx).
        """
        pts, cell_area = self._ensure_grid(area_xy)
        P = jnp.asarray(positions_xy, dtype=float).reshape(-1, 2)
        n = int(P.shape[0])
        phi = build_phi_grid(pts, centers, weights, self.cfg.bump_sigma, self.cfg.bump_floor)
        labels, M, C, cost = partition_centroids_cost(P, phi, pts, cell_area, n)

        if self.cfg.step_mode == "lloyd":
            new = C
        else:                                # the Lyapunov-tested formula, shared code
            new = capped_descent(P, C, self.cfg.gain, float(dt), self.cfg.v_max)

        xmn, xmx, ymn, ymx = (float(a) for a in area_xy)
        new = jnp.stack([jnp.clip(new[:, 0], xmn, xmx),
                         jnp.clip(new[:, 1], ymn, ymx)], axis=1)
        return {
            "new_xy": np.asarray(new),
            "labels": np.asarray(labels).reshape(self._ny, self._nx),
            "centroids": np.asarray(C),
            "cost": float(cost),
            "phi": np.asarray(phi).reshape(self._ny, self._nx),
        }
