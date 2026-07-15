"""BEV (bird's-eye-view) scene visualization for the tracking sim.

Renders the standard multi-object-tracking scene from a ``run_tracking_episode``
result: drone cameras + ground footprints, ground-truth target trails, detections,
and GM-PHD track estimates with covariance ellipses. Outputs an animated GIF or a
single-frame PNG. Pure matplotlib (Agg backend) — no heavy tracking-framework
dependency; Stone Soup remains the reference for richer plotting.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                       # headless rendering to file

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import chi2


def covariance_ellipse(cov, confidence: float = 0.95,
                       n_std: Optional[float] = None) -> Tuple[float, float, float]:
    """Ellipse (full width, full height, angle in degrees) for a 2x2 covariance.

    Standard eigendecomposition recipe: axes are 2 * scale * sqrt(eigenvalue),
    oriented along the major eigenvector. ``confidence`` sets the scale via the
    chi-square quantile (95% -> ~2.45 sigma in 2D); pass ``n_std`` to override.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError(f"covariance must be 2x2, got shape {cov.shape}")
    if not np.all(np.isfinite(cov)):
        return (0.0, 0.0, 0.0)                       # degenerate/NaN cov -> no ellipse
    vals, vecs = np.linalg.eigh(cov)                 # ascending eigenvalues
    order = np.argsort(vals)[::-1]                   # descending: major first
    vals, vecs = vals[order], vecs[:, order]
    if n_std is None:
        n_std = float(np.sqrt(chi2.ppf(confidence, 2)))
    # non-positive eigenvalues (non-PSD input) are clamped to 0 -> degenerate axis
    width = 2.0 * n_std * float(np.sqrt(max(vals[0], 0.0)))
    height = 2.0 * n_std * float(np.sqrt(max(vals[1], 0.0)))
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return width, height, angle


_LEGEND = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor="tab:blue", markersize=9,
           label="drone (camera)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=8,
           label="target (truth)"),
    Line2D([0], [0], marker="x", color="tab:cyan", linestyle="None", markersize=7,
           label="detection"),
    Line2D([0], [0], marker=".", color="tab:red", markersize=12, linestyle="None",
           label="GM-PHD track"),
    Line2D([0], [0], color="tab:red", label="95% covariance"),
]


def _draw_bev(ax, frames, i, area_xy, fov, confidence, target_hist):
    xmn, xmx, ymn, ymx = area_xy
    ax.clear()
    ax.set_xlim(xmn, xmx)
    ax.set_ylim(ymn, ymx)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    f = frames[i]

    # drones + camera footprints
    for d in np.asarray(f["drones"], dtype=float):
        if fov is not None and len(d) >= 3:
            r = float(d[2]) * np.tan(fov)
            ax.add_patch(Circle((d[0], d[1]), r, fill=False, ec="tab:blue",
                                alpha=0.20, ls="--", lw=0.8, zorder=1))
        ax.plot(d[0], d[1], "^", color="tab:blue", ms=9, zorder=4)

    # ground-truth target trails + current positions
    tgt = np.asarray(f["targets"], dtype=float)
    for m in range(tgt.shape[0]):
        trail = np.array([np.asarray(target_hist[k], dtype=float)[m] for k in range(i + 1)])
        ax.plot(trail[:, 0], trail[:, 1], "-", color="0.6", lw=1.0, alpha=0.7, zorder=2)
        ax.plot(tgt[m, 0], tgt[m, 1], "o", color="black", ms=8, zorder=5)

    # detections
    for (z, _R) in f["detections"]:
        z = np.asarray(z, dtype=float)
        ax.plot(z[0], z[1], "x", color="tab:cyan", ms=7, mew=1.6, zorder=3)

    # GM-PHD track estimates + covariance ellipses
    for (pos, P, _w) in f["estimates"]:
        pos = np.asarray(pos, dtype=float)
        width, height, angle = covariance_ellipse(P, confidence=confidence)
        ax.add_patch(Ellipse(xy=(pos[0], pos[1]), width=width, height=height,
                             angle=angle, fill=False, ec="tab:red", lw=1.8, zorder=6))
        ax.plot(pos[0], pos[1], ".", color="tab:red", ms=12, zorder=7)

    ax.set_title(f"t = {f['t']:>3d}     targets = {tgt.shape[0]}     "
                 f"GM-PHD tracks = {f['n_estimates']}")
    ax.legend(handles=_LEGEND, loc="upper right", fontsize=8, framealpha=0.9)


def _resolve(result, area_xy, fov):
    area = area_xy if area_xy is not None else result.get("area_xy")
    if area is None:
        raise ValueError("area_xy not provided and not present in result")
    f = fov if fov is not None else result.get("sensor_half_fov_rad")
    frames = result["frames"]
    # target trails assume a stable per-target row order; enforce constant count
    n0 = int(np.asarray(frames[0]["targets"]).shape[0])
    if any(int(np.asarray(fr["targets"]).shape[0]) != n0 for fr in frames):
        raise ValueError("target count must be constant across frames for trail rendering")
    target_hist = [fr["targets"] for fr in frames]
    return area, f, target_hist


def animate_tracking(result: Dict, out_path: str, area_xy=None,
                     sensor_half_fov_rad: Optional[float] = None, fps: int = 5,
                     dpi: int = 110, confidence: float = 0.95,
                     figsize: Tuple[float, float] = (8, 8)) -> str:
    """Render the tracking run as an animated GIF (one frame per step)."""
    frames = result["frames"]
    if not frames:
        raise ValueError("no frames to animate")
    area, fov, target_hist = _resolve(result, area_xy, sensor_half_fov_rad)
    fig, ax = plt.subplots(figsize=figsize)

    def animate(i):
        _draw_bev(ax, frames, i, area, fov, confidence, target_hist)

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000.0 / fps)
    try:
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(fig)                               # never leak the figure
    return out_path


def plot_frame(result: Dict, out_path: str, frame_idx: int = -1, area_xy=None,
               sensor_half_fov_rad: Optional[float] = None, confidence: float = 0.95,
               figsize: Tuple[float, float] = (8, 8), dpi: int = 130) -> str:
    """Render a single frame (default: last) as a PNG for a quick look."""
    frames = result["frames"]
    if not frames:
        raise ValueError("no frames to plot")
    i = frame_idx % len(frames)
    area, fov, target_hist = _resolve(result, area_xy, sensor_half_fov_rad)
    fig, ax = plt.subplots(figsize=figsize)
    _draw_bev(ax, frames, i, area, fov, confidence, target_hist)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


def drone_view_image(frame, area, res: int = 96, show_coverage: bool = True,
                     belief_gain: float = 2.2, belief_sigma_floor: float = 8.0):
    """Render the DRONE VIEW (what the filter perceives) as an (res, res, 4) RGBA float
    image in [0, 1], row 0 = ymin.

    Two layers, as confirmed: a coverage-certainty background (deep blue = uncertain
    -> red = certain) plus the continuous GM-PHD belief as summed Gaussians overlaid
    as bright (yellow/white) hotspots. Tight+bright = confident target; broad+dim =
    uncertain. No 0.5 threshold, so nothing flickers.
    """
    from scipy.ndimage import zoom

    xmn, xmx, ymn, ymx = area
    res = int(res)
    img = np.zeros((res, res, 4), dtype=float)
    img[..., 3] = 1.0

    # coverage background, upsampled smoothly to the texture resolution
    if show_coverage and "coverage" in frame:
        cov = np.asarray(frame["coverage"], dtype=float)
        ny, nx = cov.shape
        cov_up = np.clip(zoom(cov, (res / ny, res / nx), order=1), 0.0, 1.0)
    else:
        cov_up = np.zeros((res, res))
    img[..., 0] = 0.10 + 0.62 * cov_up
    img[..., 1] = 0.16 + 0.10 * cov_up
    img[..., 2] = 0.55 - 0.40 * cov_up

    # continuous GM-PHD belief intensity (sum of weighted Gaussians)
    xs = np.linspace(xmn, xmx, res)
    ys = np.linspace(ymn, ymx, res)
    gx, gy = np.meshgrid(xs, ys)
    floor = (belief_sigma_floor ** 2) * np.eye(2)
    inten = np.zeros((res, res))
    for (pos, P, w) in frame.get("belief", []):
        # inflate the covariance for DISPLAY so even a confident (tight) track shows
        # as a visible blob instead of a single sub-cell point.
        Pv = np.asarray(P, dtype=float) + floor
        try:
            Pinv = np.linalg.inv(Pv)
        except np.linalg.LinAlgError:
            continue
        d0 = gx - pos[0]
        d1 = gy - pos[1]
        md = Pinv[0, 0] * d0 * d0 + 2 * Pinv[0, 1] * d0 * d1 + Pinv[1, 1] * d1 * d1
        inten += float(w) * np.exp(-0.5 * md)
    inten = np.clip(inten * belief_gain, 0.0, 1.0)

    # belief -> WHITE-HOT core (pushes all channels up) so it pops as a bright spot
    # against the red coverage; the soft fade back to red reads as the belief spread.
    img[..., 0] = np.clip(img[..., 0] + 0.95 * inten, 0.0, 1.0)
    img[..., 1] = np.clip(img[..., 1] + 0.92 * inten, 0.0, 1.0)
    img[..., 2] = np.clip(img[..., 2] + 0.88 * inten, 0.0, 1.0)
    return img
