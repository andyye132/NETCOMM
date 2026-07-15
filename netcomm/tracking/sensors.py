"""Downward-facing camera sensor model: ground-plane back-projection to (z, R).

A drone at 3D position (dx, dy, h) with a nadir-pointing camera observes ground
targets on the z=0 plane. The image-plane detection is back-projected to the
ground plane (giving position z = target xy under a flat-ground assumption), and
the measurement covariance R is obtained by propagating pixel-localization and
altitude/pose noise through the projection geometry.

First-order covariance model. For a target at horizontal offset rho from the
drone's nadir point: off-nadir angle alpha = atan2(rho, h), slant range
s = hypot(rho, h), bearing phi from nadir to target.
    sigma_t (across-track) = (s / f) * sigma_px
    sigma_r (along-track)  = (s / f) * sigma_px / cos(alpha) + tan(alpha) * sigma_alt
    R = Rot(phi) diag(sigma_r^2, sigma_t^2) Rot(phi)^T
=> isotropic directly overhead; grows with altitude and off-nadir angle; elongated
radially (toward/away from the drone). Fusing drones from different bearings
shrinks the ellipse. This is the camera analogue of a range/bearing sensor and
matches ModTrack's camera front-end.

The geometry (per drone x target footprint mask, projected z, covariance R) is a
genuine-JAX kernel: ``jnp`` math ``vmap``-ed over the drones x targets grid and
``jit``-compiled. ``simulate_detections`` evaluates that kernel once, then
materializes the surviving (detected) pairs into the Python list of
``gmphd.Detection`` objects the tracker consumes (detection count is
data-dependent, so the list boundary stays in Python).

Produces gmphd.Detection (z, R) — the interface the tracker consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from gmphd.types import Detection


@dataclass
class CameraSensorConfig:
    focal_px: float = 600.0                 # focal length in pixels
    sigma_px: float = 1.5                   # pixel-localization std (detector noise)
    sigma_alt: float = 0.5                  # altitude/pose std (m)
    half_fov_rad: float = np.deg2rad(35.0)  # half field-of-view -> ground footprint radius
    p_detect: float = 0.95                  # detection prob for an in-footprint target (LoS link)
    clutter_rate: float = 0.0               # expected false detections per drone per step
    min_altitude: float = 1.0               # drones below this altitude are not cameras
    min_cov: float = 1e-4                   # floor on covariance eigenvalues (m^2)

    # --- optional LoS/NLoS elevation-dependent sensing ---
    # When off, the model is the plain geometric camera above. When on, each look is
    # LoS w.p. P_LoS(theta), else NLoS (inflated covariance + lower detection prob).
    # The A2G LoS-probability sigmoid is Al-Hourani et al. 2014, as adopted in
    # Alzenad & Yanikomeroglu 2018, "Coverage and Rate Analysis for UAV Base Stations
    # with LoS/NLoS Propagation", Eq (1):
    #     P_L(z) = 1 / (1 + a*exp(-b*((180/pi)*atan(h/z) - a)))
    # where (180/pi)*atan(h/z) is the elevation angle (deg) and (a, b) are environment
    # constants. Defaults are the DENSE-URBAN values used in that paper's Sec. VI
    # (a=12.08, b=0.11). Other Al-Hourani environments: suburban (4.88, 0.43),
    # urban (9.61, 0.16), high-rise urban (27.23, 0.08).
    los_nlos: bool = False
    los_a: float = 12.08                    # LoS sigmoid 'a'; dense-urban (Alzenad 2018 Sec VI)
    los_b: float = 0.11                     # LoS sigmoid 'b'; dense-urban
    # NLoS sensing degradation: a modeling assumption for THIS camera sensor (not from the
    # cited paper, which models RF coverage/rate, not detection covariance). An NLoS look
    # inflates the LoS covariance R and lowers the detection probability.
    nlos_R_scale: float = 6.0               # NLoS inflates the LoS covariance R by this factor
    p_detect_nlos: float = 0.4              # detection prob on an NLoS link (vs p_detect on LoS)


def camera_measurement(drone_xyz, target_xy, cfg: CameraSensorConfig
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Noiseless ground projection (= target_xy) and its (2x2) measurement covariance R.

    Kept as exact float64 NumPy: this scalar projection is a known-answer reference
    (the geometry unit tests pin it to ~1e-12) and is reused by the PCRLB/planner
    information terms. The batched sensing hot loop is JAX-vectorized separately in
    ``_detection_grid_kernel`` below."""
    dx, dy, h = float(drone_xyz[0]), float(drone_xyz[1]), float(drone_xyz[2])
    tx, ty = float(target_xy[0]), float(target_xy[1])
    rx, ry = tx - dx, ty - dy
    rho = float(np.hypot(rx, ry))
    h_eff = max(h, 1e-6)
    s = float(np.hypot(rho, h_eff))
    alpha = float(np.arctan2(rho, h_eff))
    cos_a = max(np.cos(alpha), 1e-3)

    sigma_t = (s / cfg.focal_px) * cfg.sigma_px
    sigma_r = (s / cfg.focal_px) * cfg.sigma_px / cos_a + np.tan(alpha) * cfg.sigma_alt

    phi = np.arctan2(ry, rx) if rho > 1e-9 else 0.0
    c, sn = np.cos(phi), np.sin(phi)
    Rot = np.array([[c, -sn], [sn, c]])
    D = np.diag([max(sigma_r ** 2, cfg.min_cov), max(sigma_t ** 2, cfg.min_cov)])
    R = Rot @ D @ Rot.T
    z = np.array([tx, ty])
    return z, R


def in_footprint(drone_xyz, target_xy, cfg: CameraSensorConfig) -> bool:
    """True if the target lies within the camera's circular ground footprint."""
    h = float(drone_xyz[2])
    rho = float(np.hypot(target_xy[0] - drone_xyz[0], target_xy[1] - drone_xyz[1]))
    return rho <= h * np.tan(cfg.half_fov_rad)


def elevation_deg(drone_xyz, target_xy) -> float:
    """Elevation angle (deg) from the target up to the drone: 90 directly overhead,
    -> 0 as the drone gets low/far. theta = atan2(h, horizontal_range)."""
    h = float(drone_xyz[2])
    rho = float(np.hypot(target_xy[0] - drone_xyz[0], target_xy[1] - drone_xyz[1]))
    return float(np.degrees(np.arctan2(h, max(rho, 1e-9))))


def los_probability(theta_deg: float, cfg: CameraSensorConfig) -> float:
    """Air-to-ground LoS probability sigmoid (Alzenad & Yanikomeroglu 2018 Eq (1),
    after Al-Hourani 2014) in the elevation angle theta = (180/pi)*atan(h/z) [deg]:
    P_LoS = 1 / (1 + a*exp(-b*(theta - a)))  -> ~1 overhead, ~0 at grazing angles.
    P_NLoS = 1 - P_LoS. (a, b) are the environment constants in CameraSensorConfig."""
    return float(1.0 / (1.0 + cfg.los_a * np.exp(-cfg.los_b * (theta_deg - cfg.los_a))))


def measurement_information(drone_xyz, target_xy, cfg: CameraSensorConfig,
                            gate: bool = True) -> np.ndarray:
    """Expected 2x2 position Fisher information E[p_D * R^{-1}] this drone contributes
    about a target at target_xy. Zero if the target is out of footprint (when gated) /
    below min altitude. With LoS/NLoS on, it is the detection-weighted mix of the LoS and
    NLoS branches: P_LoS*p_D*R_LoS^{-1} + P_NLoS*p_D,NLoS*R_NLoS^{-1}. This is the
    per-sensor measurement-information term H^T R^{-1} H (position block) used by the
    PCRLB bound and the MI planner.

    gate=True (default): hard footprint cutoff -> 0 outside (the sim's detection gate;
    used by the PCRLB bound which scores realized sensing). gate=False: skip the
    footprint cutoff so the information decays SMOOTHLY with distance (R grows with slant
    range) but is never exactly 0 -- the gradient an information-driven planner (greedy
    MI) needs to steer toward targets, analogous to the paper's range sensor.

    Note: the p_D weighting is the information-reduction-factor approximation to the
    PCRLB under imperfect detection (optimistic/tight when p_D < 1); clutter is ignored."""
    if float(drone_xyz[2]) < cfg.min_altitude:
        return np.zeros((2, 2))
    if gate and not in_footprint(drone_xyz, target_xy, cfg):
        return np.zeros((2, 2))
    _, R = camera_measurement(drone_xyz, target_xy, cfg)
    if not cfg.los_nlos:
        return cfg.p_detect * np.linalg.inv(R)
    pl = los_probability(elevation_deg(drone_xyz, target_xy), cfg)
    R_inv = np.linalg.inv(R)
    R_nlos_inv = np.linalg.inv(cfg.nlos_R_scale * R)
    return pl * cfg.p_detect * R_inv + (1.0 - pl) * cfg.p_detect_nlos * R_nlos_inv


@jax.jit
def _detection_grid_kernel(drones_xyz, targets_xy, focal_px, sigma_px, sigma_alt,
                           min_cov, tan_fov, min_altitude, los_a, los_b):
    """Vectorized (drones x targets) sensing geometry as pure jnp (vmap x vmap, jit).

    Returns, all stacked over (N_drones, M_targets[, ...]):
      mask : (N, M) bool  — drone airborne AND target in footprint.
      z    : (N, M, 2)    — noiseless projected position.
      R    : (N, M, 2, 2) — LoS (clean) measurement covariance.
      p_los: (N, M)       — LoS probability (Alzenad 2018 Eq (1) elevation sigmoid).
    The host samples detection/link condition from these and builds the Detection list.
    """
    def per_pair(d, t):
        dx, dy, h = d[0], d[1], d[2]
        tx, ty = t[0], t[1]
        rx, ry = tx - dx, ty - dy
        rho = jnp.hypot(rx, ry)
        h_eff = jnp.maximum(h, 1e-6)
        s = jnp.hypot(rho, h_eff)
        alpha = jnp.arctan2(rho, h_eff)
        cos_a = jnp.maximum(jnp.cos(alpha), 1e-3)
        sigma_t = (s / focal_px) * sigma_px
        sigma_r = (s / focal_px) * sigma_px / cos_a + jnp.tan(alpha) * sigma_alt
        phi = jnp.where(rho > 1e-9, jnp.arctan2(ry, rx), 0.0)
        c, sn = jnp.cos(phi), jnp.sin(phi)
        Rot = jnp.array([[c, -sn], [sn, c]])
        D = jnp.diag(jnp.array([jnp.maximum(sigma_r ** 2, min_cov),
                                jnp.maximum(sigma_t ** 2, min_cov)]))
        R = Rot @ D @ Rot.T
        z = jnp.array([tx, ty])

        in_fp = rho <= h * tan_fov
        airborne = h >= min_altitude
        mask = jnp.logical_and(in_fp, airborne)
        theta = jnp.degrees(jnp.arctan2(h, jnp.maximum(rho, 1e-9)))
        p_los = 1.0 / (1.0 + los_a * jnp.exp(-los_b * (theta - los_a)))
        return mask, z, R, p_los

    per_drone = jax.vmap(lambda d: jax.vmap(lambda t: per_pair(d, t))(targets_xy))
    return per_drone(drones_xyz)


def simulate_detections(drone_positions, target_positions, cfg: CameraSensorConfig,
                        rng: np.random.Generator,
                        area: Optional[Sequence[float]] = None) -> List[Detection]:
    """Generate noisy (z, R) detections from all drones sensing all targets.

    drone_positions: (N, 3) airborne sensor poses.
    target_positions: (M, 2) ground-truth target positions.
    area: (xmin, xmax, ymin, ymax) for uniform clutter (if clutter_rate > 0).

    The footprint mask / projected z / covariance R are computed once as a jitted jnp
    grid over (drones x targets); the surviving detections are materialized into a
    Python list at the boundary (the tracker consumes a list; the count is data-dependent).
    """
    dets: List[Detection] = []
    drones = np.asarray(drone_positions, dtype=float).reshape(-1, 3)
    targets = np.asarray(target_positions, dtype=float).reshape(-1, 2)

    if drones.shape[0] > 0 and targets.shape[0] > 0:
        tan_fov = float(np.tan(cfg.half_fov_rad))
        mask, z, R, p_los = _detection_grid_kernel(
            jnp.asarray(drones), jnp.asarray(targets),
            float(cfg.focal_px), float(cfg.sigma_px), float(cfg.sigma_alt),
            float(cfg.min_cov), tan_fov, float(cfg.min_altitude),
            float(cfg.los_a), float(cfg.los_b))
        mask = np.asarray(mask)
        z = np.asarray(z)
        R = np.asarray(R)
        p_los = np.asarray(p_los)
    else:
        mask = np.zeros((drones.shape[0], targets.shape[0]), dtype=bool)
        z = R = p_los = None

    # Iterate drone-then-target so the host RNG draw order (and any per-drone clutter)
    # is identical to the original scalar loop.
    for i, d in enumerate(drones):
        if d[2] < cfg.min_altitude:
            continue
        for j in range(targets.shape[0]):
            if not mask[i, j]:
                continue
            z0, R_los = z[i, j], R[i, j]
            if cfg.los_nlos:                        # sample the link condition by elevation
                if rng.random() < p_los[i, j]:
                    pd, R_use = cfg.p_detect, R_los                          # LoS: clean view
                else:
                    pd, R_use = cfg.p_detect_nlos, cfg.nlos_R_scale * R_los  # NLoS: occluded
            else:
                pd, R_use = cfg.p_detect, R_los
            if rng.random() > pd:                   # missed detection
                continue
            noise = rng.multivariate_normal(np.zeros(2), R_use)
            dets.append(Detection(z=z0 + noise, R=R_use))
        if cfg.clutter_rate > 0.0 and area is not None:
            for _ in range(int(rng.poisson(cfg.clutter_rate))):
                zc = np.array([rng.uniform(area[0], area[1]), rng.uniform(area[2], area[3])])
                _, Rc = camera_measurement(d, zc, cfg)
                dets.append(Detection(z=zc, R=Rc))
    return dets
