"""The GM-PHD filter recursion (Vo & Ma 2006), implemented in genuine JAX.

The multi-target posterior is represented as a Gaussian mixture (the PHD
intensity). Each step:

    predict   : survived components (p_S, F, Q) + birth (measurement- or
                intensity-driven, per ``cfg.birth_mode``)
    update    : missed-detection terms (1 - p_D) + per-measurement detection terms
    prune     : drop low-weight components
    merge     : moment-match nearby components (Vo & Ma Table II)
    cap        : keep the J_max heaviest
    extract   : report components with weight > threshold

Measurements carry per-measurement covariance R, so the innovation
S = H P H^T + R is computed per (component, measurement) pair.

Implementation notes (genuine JAX):
  - The numeric work lives in ``kernels.py`` as ``jit``/``vmap``/``scan`` kernels
    over fixed-capacity padded arrays (means [J,4], covs [J,4,4], weights [J],
    plus a validity mask). predict / Kalman update / mvn_pdf / prune / merge / cap
    are all vectorized; only the small Python orchestration and the
    list<->array marshalling at the public boundary stay in NumPy.
  - The public boundary still speaks ``GaussianComponent`` lists so the
    ``GMPHDFilter`` / ``GMPHDConfig`` / ``Detection`` / ``TargetEstimate`` API and
    ``self.components`` are byte-for-byte compatible with the rest of the sim.
  - The innovation covariance S is run through a symmetrize + eigenvalue-floor
    PSD projection (``kernels.mvn_pdf_jax`` / ``_single_update``) so a numerically
    non-PSD S degrades gracefully instead of raising.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import jax.numpy as jnp

from . import kernels
from .config import GMPHDConfig
from .models import cv_model, measurement_matrix
from .types import Detection, GaussianComponent, TargetEstimate


def _pack(components: Sequence[GaussianComponent]):
    """List[GaussianComponent] -> (means (J,4), covs (J,4,4), weights (J,))."""
    J = len(components)
    if J == 0:
        return (np.zeros((0, 4)), np.zeros((0, 4, 4)), np.zeros((0,)))
    means = np.stack([np.asarray(c.m, float).reshape(4) for c in components])
    covs = np.stack([np.asarray(c.P, float).reshape(4, 4) for c in components])
    weights = np.array([float(c.w) for c in components])
    return means, covs, weights


def _unpack(means, covs, weights, mask=None) -> List[GaussianComponent]:
    """Padded arrays -> List[GaussianComponent], keeping only valid slots."""
    means = np.asarray(means)
    covs = np.asarray(covs)
    weights = np.asarray(weights)
    n = means.shape[0]
    if mask is None:
        idx = range(n)
    else:
        mask = np.asarray(mask)
        idx = [i for i in range(n) if bool(mask[i])]
    return [GaussianComponent(w=float(weights[i]), m=means[i], P=covs[i]) for i in idx]


class GMPHDFilter:
    def __init__(self, config: GMPHDConfig | None = None):
        self.cfg = config or GMPHDConfig()
        self.F, self.Q = cv_model(self.cfg.dt, self.cfg.q)
        self.H = measurement_matrix()
        self.components: List[GaussianComponent] = []
        # JAX device copies of the constant model matrices (used by the kernels)
        self._Fj = jnp.asarray(self.F)
        self._Qj = jnp.asarray(self.Q)
        self._Hj = jnp.asarray(self.H)

    # ------------------------------------------------------------------ predict
    def predict(self, components: Sequence[GaussianComponent]) -> List[GaussianComponent]:
        """Propagate surviving components through the CV model (JAX kernel)."""
        if len(components) == 0:
            return []
        means, covs, weights = _pack(components)
        nm, nc, nw = kernels.predict_kernel(
            jnp.asarray(means), jnp.asarray(covs), jnp.asarray(weights),
            self._Fj, self._Qj, float(self.cfg.p_survival))
        return _unpack(nm, nc, nw)

    def birth(self, detections: Sequence[Detection]) -> List[GaussianComponent]:
        """Spawn birth components according to ``cfg.birth_mode``.

        'measurement' (default): one birth component per detection, centered at the
        measurement with the measurement's covariance and a broad velocity prior.
        'intensity': the fixed Vo & Ma spontaneous-birth GM intensity gamma_k
        (paper Eq. 17), independent of the detections.
        """
        cfg = self.cfg
        if cfg.birth_mode == "intensity":
            out: List[GaussianComponent] = []
            for (w, m, P) in cfg.birth_intensity:
                out.append(GaussianComponent(
                    w=float(w),
                    m=np.asarray(m, float).reshape(4),
                    P=np.asarray(P, float).reshape(4, 4)))
            return out
        # default: measurement-driven birth
        out = []
        for det in detections:
            m = np.array([det.z[0], det.z[1], 0.0, 0.0])
            P = np.zeros((4, 4))
            P[:2, :2] = det.R
            P[2, 2] = cfg.birth_vel_var
            P[3, 3] = cfg.birth_vel_var
            out.append(GaussianComponent(w=cfg.birth_weight, m=m, P=P))
        return out

    # ------------------------------------------------------------------- update
    def update(self, predicted: Sequence[GaussianComponent],
               detections: Sequence[Detection]) -> List[GaussianComponent]:
        """Full GM-PHD measurement update with per-measurement covariance (JAX)."""
        pD = float(self.cfg.p_detect)
        kappa = float(self.cfg.clutter_intensity)
        J = len(predicted)
        if J == 0:
            return []
        means, covs, weights = _pack(predicted)

        # missed-detection terms always present (one per predicted component)
        if len(detections) == 0:
            miss = [GaussianComponent(w=(1.0 - pD) * float(weights[i]),
                                      m=means[i], P=covs[i]) for i in range(J)]
            return miss

        zs = np.stack([np.asarray(d.z, float).reshape(2) for d in detections])
        Rs = np.stack([np.asarray(d.R, float).reshape(2, 2) for d in detections])
        mask = np.ones(J, dtype=bool)
        det_mask = np.ones(len(detections), dtype=bool)

        om, oc, ow, omask = kernels.update_kernel(
            jnp.asarray(means), jnp.asarray(covs), jnp.asarray(weights),
            jnp.asarray(mask), jnp.asarray(zs), jnp.asarray(Rs),
            jnp.asarray(det_mask), self._Hj, pD, kappa)
        return _unpack(om, oc, ow, omask)

    # ----------------------------------------------------- component management
    def prune(self, components: Sequence[GaussianComponent]) -> List[GaussianComponent]:
        T = float(self.cfg.prune_threshold)
        if len(components) == 0:
            return []
        means, covs, weights = _pack(components)
        mask = np.ones(len(components), dtype=bool)
        keep = np.asarray(kernels.prune_mask(
            jnp.asarray(weights), jnp.asarray(mask), T))
        return _unpack(means, covs, weights, keep)

    def merge(self, components: Sequence[GaussianComponent]) -> List[GaussianComponent]:
        """Greedily merge components within Mahalanobis^2 distance U via moment
        matching (Vo & Ma 2006, Table II). The merge gate uses each CANDIDATE
        component's own covariance P_i, not the seed's (the verified bug fix)."""
        U = float(self.cfg.merge_threshold)
        J = len(components)
        if J == 0:
            return []
        means, covs, weights = _pack(components)
        mask = np.ones(J, dtype=bool)
        # at most J merge iterations are ever needed (each removes >=1 component)
        m_out, P_out, w_out, valid_out = kernels.merge_kernel(
            jnp.asarray(means), jnp.asarray(covs), jnp.asarray(weights),
            jnp.asarray(mask), J, U)
        return _unpack(m_out, P_out, w_out, valid_out)

    def cap(self, components: Sequence[GaussianComponent]) -> List[GaussianComponent]:
        J = self.cfg.max_components
        if len(components) <= J:
            return list(components)
        means, covs, weights = _pack(components)
        mask = np.ones(len(components), dtype=bool)
        idx = np.asarray(kernels.cap_indices(
            jnp.asarray(weights), jnp.asarray(mask), J))
        return [GaussianComponent(w=float(weights[i]), m=means[i], P=covs[i])
                for i in idx]

    # ------------------------------------------------------------------ extract
    def extract(self, components: Sequence[GaussianComponent] | None = None
                ) -> List[TargetEstimate]:
        """Multi-target state extraction (Vo & Ma 2006, Table III).

        Components with weight > threshold are reported round(w) times: a merged
        component with w ~= 2 IS two coincident targets (e.g. at a track
        crossing), and the paper emits its mean once per unit of mass. The PHD
        mass is split evenly across the copies (w/n each) so downstream
        consumers (phi, target priors) stay mass-preserving — the paper's table
        defines only the state list, not per-copy weights."""
        comps = self.components if components is None else components
        thr = self.cfg.extract_threshold
        out: List[TargetEstimate] = []
        for c in comps:
            if c.w > thr:
                n = max(1, int(round(c.w)))
                for _ in range(n):
                    out.append(TargetEstimate(m=c.m, P=c.P, w=c.w / n))
        return out

    @property
    def cardinality(self) -> float:
        """Expected number of targets = total PHD mass."""
        return float(sum(c.w for c in self.components))

    # --------------------------------------------------------------------- step
    def step(self, detections: Sequence[Detection]) -> List[TargetEstimate]:
        """Run one full predict-update-manage recursion and return target estimates."""
        predicted = self.predict(self.components) + self.birth(detections)
        updated = self.update(predicted, detections)
        managed = self.cap(self.merge(self.prune(updated)))
        self.components = managed
        return self.extract(managed)
