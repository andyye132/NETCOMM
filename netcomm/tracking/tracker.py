"""Pluggable tracker interface and the toggle.

A ``Tracker`` ingests a per-step set of (z, R) detections and returns a list of
target estimates. ``make_tracker`` is the toggle: 'none' (tracking off), 'gmphd'
(the standalone GM-PHD filter), and 'modtrack' (reserved for later).
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, runtime_checkable

from gmphd import GMPHDConfig, GMPHDFilter
from gmphd.types import Detection, TargetEstimate


@runtime_checkable
class Tracker(Protocol):
    name: str

    def step(self, detections: Sequence[Detection]) -> List[TargetEstimate]:
        ...


class GMPHDTracker:
    """Adapter wrapping the standalone GM-PHD filter behind the Tracker interface."""
    name = "gmphd"

    def __init__(self, config: Optional[GMPHDConfig] = None):
        self.filter = GMPHDFilter(config or GMPHDConfig())

    def step(self, detections: Sequence[Detection]) -> List[TargetEstimate]:
        return self.filter.step(list(detections))

    @property
    def cardinality(self) -> float:
        return self.filter.cardinality

    def belief(self):
        """The full GM-PHD belief as (position, 2x2 cov, weight) per component.

        This is the continuous mixture the drones perceive — used to render the
        drone-view intensity (unlike ``step`` which returns only thresholded tracks).
        """
        return [(c.m[:2].copy(), c.P[:2, :2].copy(), float(c.w))
                for c in self.filter.components]


def make_tracker(name: Optional[str], gmphd_config: Optional[GMPHDConfig] = None):
    """Toggle a tracker by name. Returns None when tracking is off."""
    key = (name or "none").lower()
    if key in ("none", "off", ""):
        return None
    if key == "gmphd":
        return GMPHDTracker(gmphd_config)
    if key == "modtrack":
        raise NotImplementedError("ModTrack tracker toggle is not implemented yet")
    raise ValueError(f"unknown tracker {name!r}; expected one of none|gmphd|modtrack")
