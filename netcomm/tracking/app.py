"""Dear PyGui desktop app for the 2D BEV tracking sim.

Layout: a terminal-style results log on the left, the BEV sim canvas in the
centre (most of the screen), and a collapsible options panel on the right.
Workflow is configure -> run -> replay: place drones and objects on the map,
set parameters, press Run, then watch the recorded run play back.

Launch:  python -m netcomm.tracking.app
(Needs a display; build_ui/draw_scene also work headlessly for smoke tests.)
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import dearpygui.dearpygui as dpg

from netcomm.tracking.sensors import CameraSensorConfig
from netcomm.tracking.runner import run_placed_tracking, run_preset_tracking
from netcomm.tracking.visualize import covariance_ellipse, drone_view_image
from gmphd import GMPHDConfig

CANVAS_W, CANVAS_H = 820, 760
DRONE_RES = 80          # drone-view field resolution (finer than before, mesh keeps it light)
PLAY_FPS = 12

_PRESET_MAP = {"Random walk": "random_walk", "Random walk (varied)": "random_walk_varied",
               "Figure-8": "figure8", "Circle": "circle", "Square": "square", "Triangle": "triangle"}

# colours (RGBA)
C_AREA = (70, 74, 86)
C_DRONE = (90, 150, 255)
C_FOOT = (90, 150, 255, 70)
C_OBJECT_FILL = (28, 30, 36)
C_OBJECT_EDGE = (225, 225, 225)
C_OBJECT_SEEN = (90, 230, 120)        # object currently inside a drone's FOV
C_TRAIL = (150, 150, 150, 120)
C_DET = (80, 200, 220)
C_TRACK = (255, 95, 95)
C_CENTROID = (255, 200, 80)        # Voronoi cell centroid the drone is steering toward
C_REPOS_LINE = (255, 200, 80, 130)


@dataclass
class AppState:
    drones: List[list] = field(default_factory=list)      # [x, y, radius]
    objects: List[list] = field(default_factory=list)     # [x, y]
    mode: str = "none"                                     # "drone" | "object" | "none"
    area: tuple = (0.0, 300.0, 0.0, 300.0)
    sensor_fov: float = CameraSensorConfig().half_fov_rad
    n_steps: int = 60
    dt: float = 0.1
    seed: int = 0
    object_speed: float = 3.0
    tracker: str = "gmphd"
    repositioner: str = "none"
    show_heat: bool = True
    coverage_decay: float = 0.3
    view_mode: str = "true"          # "true" (reality) | "drone" (belief)
    object_mode: str = "manual"      # "manual" (placed) | "preset" (deployed path family)
    deployed: Optional[dict] = None  # {preset, n, speed, preview:[(x,y),...]}
    result: Optional[dict] = None
    frame_idx: int = 0
    playing: bool = False
    log_tags: List[int] = field(default_factory=list)
    _last: float = 0.0
    _pending_play: bool = False      # set by a run worker thread; UI thread starts playback


STATE = AppState()


# --------------------------------------------------------------------------- geometry
def _transform(area, W, H, pad=16):
    xmn, xmx, ymn, ymx = area
    scale = min((W - 2 * pad) / (xmx - xmn), (H - 2 * pad) / (ymx - ymn))
    ox = pad + ((W - 2 * pad) - scale * (xmx - xmn)) / 2.0
    oy = pad + ((H - 2 * pad) - scale * (ymx - ymn)) / 2.0
    return scale, ox, oy, xmn, ymn


def _w2s(wx, wy, tf, H):
    scale, ox, oy, xmn, ymn = tf
    return ox + (wx - xmn) * scale, H - (oy + (wy - ymn) * scale)


def _s2w(sx, sy, tf, H):
    scale, ox, oy, xmn, ymn = tf
    return xmn + (sx - ox) / scale, ymn + ((H - sy) - oy) / scale


def _ellipse_pts(pos, P, tf, H, confidence=0.95, n=28):
    width, height, angle = covariance_ellipse(P, confidence=confidence)
    if width <= 0 or height <= 0:
        return []
    a, b, th = width / 2.0, height / 2.0, np.radians(angle)
    pts = []
    for k in range(n):
        t = 2 * np.pi * k / n
        ex, ey = a * np.cos(t), b * np.sin(t)
        wx = pos[0] + ex * np.cos(th) - ey * np.sin(th)
        wy = pos[1] + ex * np.sin(th) + ey * np.cos(th)
        pts.append(_w2s(wx, wy, tf, H))
    return pts


# --------------------------------------------------------------------------- drawing
def _heat_color(c):
    c = float(np.clip(c, 0.0, 1.0))
    # blue (uncertain) -> red (certain)
    return (int(40 + 175 * c), int(95 - 35 * c), int(205 - 150 * c), 200)


def _draw_heatmap(cov, area, tf, H):
    ny, nx = cov.shape
    xmn, xmx, ymn, ymx = area
    cw = (xmx - xmn) / nx
    ch = (ymx - ymn) / ny
    for iy in range(ny):
        wy0 = ymn + iy * ch
        for ix in range(nx):
            wx0 = xmn + ix * cw
            s0 = _w2s(wx0, wy0, tf, H)
            s1 = _w2s(wx0 + cw, wy0 + ch, tf, H)
            col = _heat_color(cov[iy, ix])
            dpg.draw_rectangle((s0[0], s1[1]), (s1[0], s0[1]), parent="canvas",
                               fill=col, color=col)


def _object_seen(o, drones, foot_is_radius, fov):
    """True if object o lies within any drone's ground footprint."""
    tan_fov = np.tan(fov)
    for d in drones:
        foot = d[2] if foot_is_radius else d[2] * tan_fov
        if np.hypot(o[0] - d[0], o[1] - d[1]) <= foot:
            return True
    return False


def _draw_drones(drones, foot_is_radius, fov, scale, tf, H):
    tan_fov = np.tan(fov)
    for d in drones:
        sx, sy = _w2s(d[0], d[1], tf, H)
        foot = d[2] if foot_is_radius else d[2] * tan_fov
        dpg.draw_circle((sx, sy), foot * scale, parent="canvas", color=C_FOOT)
        dpg.draw_circle((sx, sy), 5, parent="canvas", fill=C_DRONE, color=C_DRONE)


def _draw_field_rects(img, area, tf, H, levels=12):
    """Draw the drone-view RGBA field cheaply via 2D greedy rectangle meshing.

    A single base rectangle paints the deep-blue 'uncertain' background. The rest
    of the field is quantized in colour and covered by maximal same-colour
    rectangles (grown right, then down) — far fewer draw items than one-per-cell,
    which is what makes playback smooth at a fine resolution."""
    ny, nx, _ = img.shape
    xmn, xmx, ymn, ymx = area
    tl = _w2s(xmn, ymx, tf, H)
    br = _w2s(xmx, ymn, tf, H)
    dpg.draw_rectangle((tl[0], tl[1]), (br[0], br[1]), parent="canvas",
                       fill=(26, 41, 140), color=(26, 41, 140))
    cw = (xmx - xmn) / nx
    ch = (ymx - ymn) / ny
    rgb = img[..., :3]
    base = np.array([0.10, 0.16, 0.55])
    q = np.clip((rgb * levels).astype(np.int16), 0, levels)
    salient = np.abs(rgb - base).max(axis=2) > 0.04
    visited = np.zeros((ny, nx), dtype=bool)
    for iy in range(ny):
        if not salient[iy].any():
            continue
        ix = 0
        while ix < nx:
            if not salient[iy, ix] or visited[iy, ix]:
                ix += 1
                continue
            q0, q1, q2 = q[iy, ix, 0], q[iy, ix, 1], q[iy, ix, 2]
            # grow right while same quantized colour, salient, unvisited
            w = 1
            while (ix + w < nx and salient[iy, ix + w] and not visited[iy, ix + w]
                   and q[iy, ix + w, 0] == q0 and q[iy, ix + w, 1] == q1
                   and q[iy, ix + w, 2] == q2):
                w += 1
            # grow down while the whole [ix:ix+w] span matches
            h = 1
            while iy + h < ny:
                qs = q[iy + h, ix:ix + w]
                if (salient[iy + h, ix:ix + w].all() and not visited[iy + h, ix:ix + w].any()
                        and (qs[:, 0] == q0).all() and (qs[:, 1] == q1).all()
                        and (qs[:, 2] == q2).all()):
                    h += 1
                else:
                    break
            visited[iy:iy + h, ix:ix + w] = True
            c = rgb[iy, ix]
            col = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255)
            s0 = _w2s(xmn + ix * cw, ymn + iy * ch, tf, H)
            s1 = _w2s(xmn + (ix + w) * cw, ymn + (iy + h) * ch, tf, H)
            dpg.draw_rectangle((s0[0], s1[1]), (s1[0], s0[1]), parent="canvas",
                               fill=col, color=col)
            ix += w


def _draw_repos(frame, tf, H):
    """Overlay the Voronoi repositioning target: each drone's cell centroid and the
    line it is steering along (drawn in both true and drone views when enabled)."""
    cents = frame.get("repos_centroids")
    if cents is None:
        return
    drns = np.asarray(frame["drones"], dtype=float)
    cents = np.asarray(cents, dtype=float)
    for k in range(min(len(drns), len(cents))):
        dsx, dsy = _w2s(drns[k][0], drns[k][1], tf, H)
        csx, csy = _w2s(cents[k][0], cents[k][1], tf, H)
        dpg.draw_line((dsx, dsy), (csx, csy), parent="canvas", color=C_REPOS_LINE, thickness=1)
        dpg.draw_circle((csx, csy), 4, parent="canvas", fill=C_CENTROID, color=C_CENTROID)


# distinct translucent fills for per-drone Voronoi cells (cycled by drone index)
_VORONOI_PALETTE = [
    (90, 150, 255), (90, 230, 120), (255, 170, 70), (220, 110, 200),
    (110, 220, 220), (235, 95, 95), (190, 200, 80), (150, 130, 245),
]


def _voronoi_fill(drone_idx, alpha=70):
    r, g, b = _VORONOI_PALETTE[int(drone_idx) % len(_VORONOI_PALETTE)]
    return (r, g, b, alpha)


def _static_voronoi_labels(drones, area, nx=64, ny=64):
    """Nearest-drone Voronoi partition (ny, nx) for the CONFIG-view preview, so the cells
    appear as soon as isotropic_voronoi is selected and drones are placed (before any Run).
    During playback the live partition in frame['repos_labels'] is used instead."""
    drones = np.asarray(drones, dtype=float)
    if drones.ndim != 2 or drones.shape[0] == 0:
        return None
    xmn, xmx, ymn, ymx = area
    xs = xmn + (np.arange(nx) + 0.5) * (xmx - xmn) / nx
    ys = ymn + (np.arange(ny) + 0.5) * (ymx - ymn) / ny
    gx, gy = np.meshgrid(xs, ys)                          # (ny, nx) world cell centres
    d2 = (gx[..., None] - drones[:, 0]) ** 2 + (gy[..., None] - drones[:, 1]) ** 2
    return np.argmin(d2, axis=-1).astype(int)            # (ny, nx) owning-drone index


def _on_repos_change(sender, app_data):
    """Selecting isotropic_voronoi auto-enables the Voronoi cell view (and redraws), so the
    user gets the cell partition without hunting for the checkbox."""
    if app_data == "isotropic_voronoi" and dpg.does_item_exist("show_voronoi_check"):
        dpg.set_value("show_voronoi_check", True)
    draw_scene(STATE)


def _draw_voronoi(frame, area, tf, H):
    """Render the live Voronoi cell partition recorded in frame['repos_labels'] — an (ny, nx)
    int grid where each cell holds the index of the drone that owns it. Cells are colour-coded
    per owning drone using the same per-cell rectangle math as _draw_heatmap, with the greedy
    horizontal run-merging trick from _draw_field_rects (grow a maximal same-owner rectangle
    along each row) so the partition stays light enough to draw at PLAY_FPS.

    No-ops gracefully when repos_labels is absent (non-Voronoi methods record no labels)."""
    labels = frame.get("repos_labels")
    if labels is None:
        return
    labels = np.asarray(labels)
    if labels.ndim != 2:
        return
    ny, nx = labels.shape
    xmn, xmx, ymn, ymx = area
    cw = (xmx - xmn) / nx
    ch = (ymx - ymn) / ny
    for iy in range(ny):
        wy0 = ymn + iy * ch
        ix = 0
        while ix < nx:
            owner = int(labels[iy, ix])
            # grow right while the same drone owns the run (greedy meshing along the row)
            w = 1
            while ix + w < nx and int(labels[iy, ix + w]) == owner:
                w += 1
            wx0 = xmn + ix * cw
            s0 = _w2s(wx0, wy0, tf, H)
            s1 = _w2s(wx0 + w * cw, wy0 + ch, tf, H)
            col = _voronoi_fill(owner)
            dpg.draw_rectangle((s0[0], s1[1]), (s1[0], s0[1]), parent="canvas",
                               fill=col, color=(col[0], col[1], col[2], 150))
            ix += w


def draw_scene(state: AppState):
    if not dpg.does_item_exist("canvas"):
        return
    dpg.delete_item("canvas", children_only=True)
    W, H = CANVAS_W, CANVAS_H
    tf = _transform(state.area, W, H)
    scale = tf[0]
    xmn, xmx, ymn, ymx = state.area
    tl = _w2s(xmn, ymx, tf, H)        # area top-left on screen
    br = _w2s(xmx, ymn, tf, H)        # area bottom-right on screen

    frame = state.result["frames"][state.frame_idx] if state.result is not None else None
    config = frame is None

    if state.view_mode == "drone" and not config:
        # DRONE VIEW — continuous GM-PHD belief + coverage drawn as a fine cell field
        boost = dpg.get_value("belief_slider") if dpg.does_item_exist("belief_slider") else 2.2
        img = drone_view_image(frame, state.area, res=DRONE_RES,
                               show_coverage=state.show_heat, belief_gain=boost)
        _draw_field_rects(img, state.area, tf, H)
        _draw_drones(frame["drones"], False, state.sensor_fov, scale, tf, H)
    else:
        # TRUE VIEW (or configuration) — reality: object points + drones
        if config:
            objects = (state.deployed["preview"]
                       if (state.object_mode == "preset" and state.deployed) else state.objects)
        else:
            objects = frame["targets"]
        if not config and state.result is not None:
            hist = state.result["frames"]
            jump = 0.4 * (xmx - xmn)
            for m in range(len(objects)):
                wp = [hist[k]["targets"][m] for k in range(state.frame_idx + 1)]
                for k in range(len(wp) - 1):
                    if np.hypot(wp[k + 1][0] - wp[k][0], wp[k + 1][1] - wp[k][1]) > jump:
                        continue          # respawn teleport — don't connect the trail
                    a_ = _w2s(wp[k][0], wp[k][1], tf, H)
                    b_ = _w2s(wp[k + 1][0], wp[k + 1][1], tf, H)
                    dpg.draw_line(a_, b_, parent="canvas", color=C_TRAIL)
        drns = state.drones if config else frame["drones"]
        for o in objects:
            sx, sy = _w2s(o[0], o[1], tf, H)
            if _object_seen(o, drns, config, state.sensor_fov):
                dpg.draw_circle((sx, sy), 6, parent="canvas", fill=C_OBJECT_SEEN, color=C_OBJECT_SEEN)
                dpg.draw_circle((sx, sy), 10, parent="canvas", color=C_OBJECT_SEEN, thickness=2)
            else:
                dpg.draw_circle((sx, sy), 6, parent="canvas", fill=C_OBJECT_FILL, color=C_OBJECT_EDGE)
        _draw_drones(drns, config, state.sensor_fov, scale, tf, H)

    show_voronoi = (dpg.get_value("show_voronoi_check")
                    if dpg.does_item_exist("show_voronoi_check") else False)
    repos_sel = dpg.get_value("repos_combo") if dpg.does_item_exist("repos_combo") else "none"
    if show_voronoi:
        if frame is not None and frame.get("repos_labels") is not None:
            _draw_voronoi(frame, state.area, tf, H)              # live partition during playback
        elif config and repos_sel == "isotropic_voronoi":
            labels = _static_voronoi_labels(state.drones, state.area)   # static preview from placed drones
            if labels is not None:
                _draw_voronoi({"repos_labels": labels}, state.area, tf, H)

    show_repos = dpg.get_value("show_repos_check") if dpg.does_item_exist("show_repos_check") else True
    if frame is not None and show_repos:
        _draw_repos(frame, tf, H)                    # gold centroid dots/lines on top

    dpg.draw_rectangle((tl[0], tl[1]), (br[0], br[1]), parent="canvas", color=C_AREA)


# --------------------------------------------------------------------------- log
def _set_status(txt):
    if dpg.does_item_exist("status_text"):
        dpg.set_value("status_text", txt)


def _clear_log():
    if dpg.does_item_exist("results_log"):
        dpg.delete_item("results_log", children_only=True)
    STATE.log_tags = []


def _append_log(line):
    if not dpg.does_item_exist("results_log"):
        return
    tag = dpg.add_text(line, parent="results_log")
    STATE.log_tags.append(tag)
    if len(STATE.log_tags) > 500:
        old = STATE.log_tags.pop(0)
        if dpg.does_item_exist(old):
            dpg.delete_item(old)
    dpg.set_y_scroll("results_log", 1e9)


def _log_frame(i):
    f = STATE.result["frames"][i]
    ests = f["estimates"]
    card = float(sum(w for (_p, _P, w) in ests))
    if ests:
        sig = float(np.mean([np.sqrt(np.trace(P) / 2.0) for (_p, P, _w) in ests]))
    else:
        sig = 0.0
    unc = 1.0 - f.get("mean_certainty", 0.0)
    _append_log(f"t{f['t']:>3}  det{len(f['detections']):>2}  trk{f['n_estimates']}  "
                f"card{card:4.1f}  unc{unc * 100:3.0f}%  s{sig:4.2f}")


def _update_metrics(i):
    """Update the live trend plots for frame i. PRIMARY (tracking quality): the object-tied
    TARGET TRACK uncertainty (mean GM-PHD track sigma, which drops as targets are tracked) plus
    a live tracked-fraction indicator (estimates vs ground-truth objects). SECONDARY: whole-map
    AREA COVERAGE (1 - mean coverage), which is NOT a tracking-quality metric and stays high
    while drones cover only part of the map."""
    if STATE.result is None or not dpg.does_item_exist("trk_headline"):
        return
    frames = STATE.result["frames"]

    # PRIMARY — target track uncertainty (mean GM-PHD track sigma)
    def _sig(f):
        ests = f["estimates"]
        return float(np.mean([np.sqrt(np.trace(P) / 2.0) for (_p, P, _w) in ests])) if ests else 0.0
    sig = [_sig(frames[k]) for k in range(i + 1)]
    dpg.set_value("trk_headline", f"{sig[-1]:.2f} m   (mean track sigma)")
    dpg.set_value("trk_plot", sig)

    # live tracked fraction: estimated tracks vs ground-truth objects this frame
    fi = frames[i]
    n_obj = int(np.asarray(fi["targets"]).reshape(-1, 2).shape[0]) if "targets" in fi else 0
    n_trk = int(fi["n_estimates"])
    frac = min(n_trk / n_obj, 1.0) if n_obj else 0.0
    if dpg.does_item_exist("trkfrac_headline"):
        dpg.set_value("trkfrac_headline", f"tracked: {n_trk}/{n_obj} objects ({frac * 100:.0f}%)")
    dpg.set_value("trkcount_plot", [float(frames[k]["n_estimates"]) for k in range(i + 1)])

    # SECONDARY — whole-map area coverage (not tracking quality)
    if dpg.does_item_exist("unc_headline"):
        cov = [1.0 - float(frames[k].get("mean_certainty", 0.0)) for k in range(i + 1)]
        dpg.set_value("unc_headline", f"{cov[-1] * 100:.0f} %   (fraction of map unobserved)")
        dpg.set_value("unc_plot", cov)


# --------------------------------------------------------------------------- callbacks
def _set_mode(mode):
    STATE.mode = "none" if STATE.mode == mode else mode
    msg = {"drone": ">>>  PLACING DRONES  -  click the map to drop a drone  <<<",
           "object": ">>>  PLACING OBJECTS  -  click the map to drop an object  <<<",
           "none": "idle"}[STATE.mode]
    _set_status(msg)
    if dpg.does_item_exist("status_text"):
        dpg.configure_item("status_text",
                           color=(255, 220, 80) if STATE.mode != "none" else (160, 200, 160))
    if dpg.does_item_exist("add_drone_btn"):
        dpg.set_item_label("add_drone_btn",
                           "* PLACING DRONES (click map) *" if STATE.mode == "drone"
                           else "Add drone (click map)")
    if dpg.does_item_exist("add_object_btn"):
        dpg.set_item_label("add_object_btn",
                           "* PLACING OBJECTS (click map) *" if STATE.mode == "object"
                           else "Add object (click map)")


def _on_canvas_click(sender=None, app_data=None):
    st = STATE
    if st.mode == "none" or not dpg.does_item_exist("canvas"):
        return
    if not dpg.is_item_hovered("canvas"):
        return
    mx, my = dpg.get_drawing_mouse_pos()
    tf = _transform(st.area, CANVAS_W, CANVAS_H)
    wx, wy = _s2w(mx, my, tf, CANVAS_H)
    xmn, xmx, ymn, ymx = st.area
    if not (xmn <= wx <= xmx and ymn <= wy <= ymx):
        return
    if st.mode == "drone":
        st.drones.append([wx, wy, float(dpg.get_value("radius_slider"))])
        dpg.set_value("drone_count", f"drones: {len(st.drones)}")
    else:
        st.objects.append([wx, wy])
        dpg.set_value("object_count", f"objects: {len(st.objects)}")
        st.object_mode = "manual"     # placing objects switches off any deployed preset
        st.deployed = None
    st.result = None                  # return to config view to show the placement
    draw_scene(st)


def _on_clear_drones():
    STATE.drones = []
    dpg.set_value("drone_count", "drones: 0")
    STATE.result = None
    draw_scene(STATE)


def _on_clear_objects():
    STATE.objects = []
    dpg.set_value("object_count", "objects: 0")
    STATE.result = None
    draw_scene(STATE)


def _on_deploy():
    """Deploy a preset path family: drop its objects onto the map (config preview)."""
    from netcomm.tracking.paths import preset_trajectories
    st = STATE
    preset = _PRESET_MAP[dpg.get_value("preset_combo")]
    n = int(dpg.get_value("preset_n_slider"))
    speed = float(dpg.get_value("preset_speed_slider"))
    traj0 = preset_trajectories(preset, n, 1, st.dt, st.area, speed, seed=st.seed)[0]  # (n, 2)
    st.object_mode = "preset"
    st.objects = []
    st.mode = "none"
    st.result = None
    st.deployed = {"preset": preset, "n": n, "speed": speed,
                   "preview": [[float(p[0]), float(p[1])] for p in traj0]}
    dpg.set_value("object_count", "objects: 0")
    draw_scene(st)
    _set_status(f"deployed {n} '{dpg.get_value('preset_combo')}' objects - press Run")


def _on_clear_preset():
    STATE.object_mode = "manual"
    STATE.deployed = None
    STATE.result = None
    draw_scene(STATE)
    _set_status("preset cleared")


def _on_toggle_heat():
    STATE.show_heat = bool(dpg.get_value("show_heat_check"))
    draw_scene(STATE)


def _on_toggle_view():
    STATE.view_mode = "drone" if STATE.view_mode == "true" else "true"
    dpg.set_item_label("view_btn", f"View: {'Drone' if STATE.view_mode == 'drone' else 'True'}")
    draw_scene(STATE)


def _on_run():
    st = STATE
    st.n_steps = int(dpg.get_value("steps_slider"))
    st.dt = float(dpg.get_value("dt_slider"))
    st.seed = int(dpg.get_value("seed_slider"))
    st.object_speed = float(dpg.get_value("speed_slider"))
    st.tracker = dpg.get_value("tracker_combo")
    st.repositioner = dpg.get_value("repos_combo")
    st.coverage_decay = float(dpg.get_value("decay_slider"))
    sensor = CameraSensorConfig()
    st.sensor_fov = sensor.half_fov_rad
    if not st.drones:
        _set_status("place at least one drone first")
        return

    drones = [tuple(dr) for dr in st.drones]
    objs = [tuple(o) for o in st.objects]
    preset = st.deployed if (st.object_mode == "preset" and st.deployed) else None

    def _prog(i, n):                                   # per-step progress (thread-safe set_value)
        if dpg.does_item_exist("run_progress"):
            dpg.set_value("run_progress", i / max(n, 1))
            dpg.configure_item("run_progress", overlay=f"{i}/{n}")

    def _worker():
        try:
            if preset is not None:
                res = run_preset_tracking(
                    drones, preset["preset"], n_objects=preset["n"], n_steps=st.n_steps, dt=st.dt,
                    area_xy=st.area, sensor_cfg=sensor, gmphd_cfg=GMPHDConfig(dt=st.dt),
                    object_speed=preset["speed"], tracker=st.tracker, repositioner=st.repositioner,
                    seed=st.seed, coverage_decay=st.coverage_decay, progress=_prog)
            else:
                res = run_placed_tracking(
                    drones, objs, n_steps=st.n_steps, dt=st.dt, area_xy=st.area, sensor_cfg=sensor,
                    gmphd_cfg=GMPHDConfig(dt=st.dt), object_speed=st.object_speed,
                    tracker=st.tracker, repositioner=st.repositioner, seed=st.seed,
                    coverage_decay=st.coverage_decay, progress=_prog)
            st.result = res
            st._pending_play = True                    # UI thread (in _tick) starts playback + drawing
        except Exception as e:                         # surface failures instead of a silent hang
            st.result = None
            _set_status(f"run failed: {type(e).__name__}: {e}")
            if dpg.does_item_exist("run_btn"):
                dpg.configure_item("run_btn", enabled=True, label="Run")
                dpg.configure_item("run_progress", show=False)

    dpg.configure_item("run_btn", enabled=False, label="Running...")
    if dpg.does_item_exist("run_progress"):
        dpg.set_value("run_progress", 0.0)
        dpg.configure_item("run_progress", show=True, overlay="0%")
    _set_status("running ... (first run compiles JAX, ~1-2 s)")
    threading.Thread(target=_worker, daemon=True).start()


def _on_play_pause():
    if STATE.result is None:
        return
    STATE.playing = not STATE.playing
    STATE._last = time.time()
    dpg.set_item_label("play_btn", "Pause" if STATE.playing else "Play")


def _on_reset():
    st = STATE
    st.result = None
    st.frame_idx = 0
    st.playing = False
    dpg.set_item_label("play_btn", "Play")
    dpg.configure_item("frame_slider", max_value=0)
    dpg.set_value("frame_slider", 0)
    _clear_log()
    if dpg.does_item_exist("trk_headline"):
        dpg.set_value("trk_headline", "--")
        dpg.set_value("trk_plot", [])
    if dpg.does_item_exist("trkfrac_headline"):
        dpg.set_value("trkfrac_headline", "tracked: --")
    if dpg.does_item_exist("unc_headline"):
        dpg.set_value("unc_headline", "--")
        dpg.set_value("unc_plot", [])
    _set_status("reset")
    draw_scene(st)


def _on_scrub(sender, app_data):
    if STATE.result is None:
        return
    STATE.playing = False
    dpg.set_item_label("play_btn", "Play")
    STATE.frame_idx = int(app_data)
    draw_scene(STATE)
    _update_metrics(STATE.frame_idx)


def _on_toggle_options():
    show = not dpg.get_item_configuration("right_panel")["show"]
    dpg.configure_item("right_panel", show=show)


# --------------------------------------------------------------------------- UI
_METHODS = ["none", "isotropic_voronoi", "greedy_mi", "rsp", "minimax"]


def _metric_tag(prefix, key):
    """Stable dpg tag for a per-tab metric checkbox."""
    return f"{prefix}_metric_{key}"


def _build_metric_checkboxes(prefix):
    """Render one checkbox per available evaluate() metric (Q6), defaulting to the current
    TOP_METRICS set. ``prefix`` namespaces the tags so Single and Batch tabs don't collide."""
    from netcomm.tracking.testing import available_metrics, TOP_METRICS
    default_on = {k for k, _, _ in TOP_METRICS}
    dpg.add_text("Metrics to display:")
    for key, disp, _ in available_metrics():
        dpg.add_checkbox(label=disp, tag=_metric_tag(prefix, key),
                         default_value=(key in default_on))


def _selected_metrics(prefix):
    """Read back the user's metric selection for ``prefix``. None (=> default TOP_METRICS)
    if the checkboxes don't exist or none are checked."""
    from netcomm.tracking.testing import available_metrics
    sel = [key for key, _, _ in available_metrics()
           if dpg.does_item_exist(_metric_tag(prefix, key)) and dpg.get_value(_metric_tag(prefix, key))]
    return sel or None


def _on_random_drones_preview(sender, app_data):
    """Slider callback: drop `app_data` random drones on the map so the layout is visible.
    (Objects stay hidden ground truth, so the random-objects slider has no preview.)"""
    st = STATE
    n = int(app_data)
    seed = int(dpg.get_value("seed_slider")) if dpg.does_item_exist("seed_slider") else st.seed
    radius = float(dpg.get_value("radius_slider")) if dpg.does_item_exist("radius_slider") else 22.0
    rng = np.random.default_rng(seed)
    xmn, xmx, ymn, ymx = st.area
    st.drones = [[float(rng.uniform(xmn, xmx)), float(rng.uniform(ymn, ymx)), radius] for _ in range(n)]
    st.deployed, st.result = None, None
    if dpg.does_item_exist("drone_count"):
        dpg.set_value("drone_count", f"drones: {len(st.drones)}")
    draw_scene(st)


def _test_cfg_from(method, tracker, n_drones, n_targets, motion_label):
    from netcomm.tracking.testing import TestConfig
    st = STATE
    steps = int(dpg.get_value("steps_slider")) if dpg.does_item_exist("steps_slider") else st.n_steps
    dt = float(dpg.get_value("dt_slider")) if dpg.does_item_exist("dt_slider") else st.dt
    return TestConfig(method=method, tracker=tracker, n_drones=n_drones, n_targets=n_targets,
                      target_motion=_PRESET_MAP.get(motion_label, "random_walk"), n_steps=steps,
                      dt=dt, area=st.area, fov_deg=float(np.degrees(st.sensor_fov)))


def _on_run_single():
    """Single Method Test: score one method over N random epochs in a background thread."""
    from netcomm.tracking.testing import run_test, run_episode, default_repos_cfg, TOP_METRICS
    st = STATE
    method, tracker = dpg.get_value("single_method"), dpg.get_value("single_tracker")
    epochs = int(dpg.get_value("single_epochs"))
    n_drones, n_targets = int(dpg.get_value("single_drones")), int(dpg.get_value("single_objects"))
    motion_label = dpg.get_value("single_motion")
    metrics = _selected_metrics("single")
    cfg = _test_cfg_from(method, tracker, n_drones, n_targets, motion_label)

    if dpg.get_value("single_watch"):                        # replay one random episode visually
        rng = np.random.default_rng(st.seed)
        xmn, xmx, ymn, ymx = st.area
        drones = [(float(rng.uniform(xmn, xmx)), float(rng.uniform(ymn, ymx)), 22.0)
                  for _ in range(n_drones)]
        st.result = run_preset_tracking(
            drones, cfg.target_motion, n_objects=n_targets, n_steps=cfg.n_steps, dt=cfg.dt,
            area_xy=st.area, sensor_cfg=CameraSensorConfig(half_fov_rad=st.sensor_fov),
            tracker=tracker, repositioner=method, repos_cfg=default_repos_cfg(method), seed=st.seed)
        st.frame_idx, st.playing, st._last = 0, True, time.time()
        dpg.configure_item("frame_slider", max_value=max(cfg.n_steps - 1, 0))
        dpg.set_value("frame_slider", 0)
        draw_scene(st)

    def _worker():
        res = run_test(cfg, n_epochs=epochs, n_workers=1, metrics=metrics,
                       progress=lambda i, n: _set_status(f"{method}: epoch {i}/{n}"))
        _append_log(f"=== SINGLE  {method}  ({res['n_epochs']} epochs, {cfg.target_motion}) ===")
        for key, disp, lower in res.get("metrics", TOP_METRICS):
            s = res["stats"][key]
            _append_log(f"  {disp:<12}{'v' if lower else '^'} {s['mean']:8.2f} +/- {s['std']:6.2f}"
                        f"  [{s['min']:.2f}, {s['max']:.2f}]")
        _set_status(f"done: {method} x {res['n_epochs']} epochs")

    _set_status(f"running: {method} x {epochs} epochs ...")
    threading.Thread(target=_worker, daemon=True).start()


def _on_run_batch_gui():
    """Batch Eval: sweep the checked methods over rounds x epochs (parallel) -> log table + CSV."""
    from netcomm.tracking.testing import run_batch, format_table
    methods = [m for m, tag in (("none", "bm_none"), ("isotropic_voronoi", "bm_voronoi"),
                                ("greedy_mi", "bm_greedy"), ("rsp", "bm_rsp"), ("minimax", "bm_minimax"))
               if dpg.get_value(tag)]
    if not methods:
        _set_status("Batch: select at least one method")
        return
    cfg = _test_cfg_from(methods[0], dpg.get_value("single_tracker") if dpg.does_item_exist("single_tracker")
                         else "gmphd", int(dpg.get_value("batch_drones")),
                         int(dpg.get_value("batch_objects")), dpg.get_value("batch_motion"))
    epochs, rounds = int(dpg.get_value("batch_epochs")), int(dpg.get_value("batch_rounds"))
    bs = int(dpg.get_value("batch_size_slider"))
    metrics = _selected_metrics("batch")

    def _worker():
        rows, csv_path = run_batch(methods, cfg, n_epochs=epochs, n_rounds=rounds, batch_size=bs,
                                   name="gui_batch", metrics=metrics,
                                   progress=lambda i, n: _set_status(f"batch round {i}/{n}"))
        for line in format_table(rows, metrics=metrics).split("\n"):
            _append_log(line)
        _append_log(f"wrote {csv_path}")
        _set_status(f"batch done: {len(methods)} methods, {rounds}x{epochs}")

    _set_status(f"running batch: {methods} ({rounds}x{epochs}) ...")
    threading.Thread(target=_worker, daemon=True).start()


def _apply_theme():
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (24, 26, 32))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (30, 33, 40))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (52, 80, 140))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (72, 112, 190))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (90, 140, 230))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (42, 46, 56))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (52, 80, 140))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
    dpg.bind_theme(theme)


def build_ui(state: AppState):
    with dpg.handler_registry():
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=_on_canvas_click)

    with dpg.window(tag="primary"):
        with dpg.group(horizontal=True):
            # LEFT — results terminal
            with dpg.child_window(width=300, tag="left_panel"):
                # PRIMARY tracking-quality signal: object-tied GM-PHD track uncertainty
                dpg.add_text("TARGET TRACK UNCERTAINTY (m)", color=(120, 200, 255))
                dpg.add_text("--", tag="trk_headline", color=(180, 230, 255))
                dpg.add_simple_plot(tag="trk_plot", default_value=(), height=80, width=-1,
                                    overlay="mean GM-PHD track sigma (lower = better tracking)")
                dpg.add_text("tracked: --", tag="trkfrac_headline", color=(150, 220, 170))
                dpg.add_simple_plot(tag="trkcount_plot", default_value=(), height=48, width=-1,
                                    overlay="# tracks vs # objects")
                dpg.add_separator()
                # SECONDARY: whole-map area coverage (NOT a tracking-quality metric)
                dpg.add_text("AREA COVERAGE (fraction unobserved)", color=(255, 150, 120))
                dpg.add_text("--", tag="unc_headline", color=(255, 200, 150))
                dpg.add_simple_plot(tag="unc_plot", default_value=(), min_scale=0.0,
                                    max_scale=1.0, height=52, width=-1,
                                    overlay="1 - mean coverage (whole map)")
                dpg.add_text("whole-map coverage, not tracking quality "
                             "(stays high while drones cover only part of the map)",
                             color=(150, 160, 175), wrap=290)
                dpg.add_separator()
                dpg.add_text("LOG", color=(140, 190, 255))
                dpg.add_child_window(tag="results_log", width=-1, height=-1)

            # CENTRE — toolbar + canvas
            with dpg.child_window(width=CANVAS_W + 24, tag="center_panel"):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Run", tag="run_btn", width=70, height=28, callback=_on_run)
                    dpg.add_button(label="Play", tag="play_btn", width=64, height=28, callback=_on_play_pause)
                    dpg.add_button(label="Reset", tag="reset_btn", width=64, height=28, callback=_on_reset)
                    dpg.add_slider_int(tag="frame_slider", min_value=0, max_value=0,
                                       width=280, callback=_on_scrub)
                    dpg.add_button(label="View: True", tag="view_btn", width=92, height=28,
                                   callback=_on_toggle_view)
                    dpg.add_button(label="Options", width=70, height=28, callback=_on_toggle_options)
                dpg.add_text("idle", tag="status_text", color=(160, 200, 160))
                dpg.add_progress_bar(tag="run_progress", default_value=0.0, width=-1, height=10,
                                     show=False, overlay="0%")
                with dpg.drawlist(width=CANVAS_W, height=CANVAS_H, tag="canvas"):
                    pass

            # RIGHT — collapsible options
            with dpg.child_window(width=340, tag="right_panel"):
                with dpg.tab_bar(tag="mode_tabs"):
                    # ---------- CUSTOM: full manual control ----------
                    with dpg.tab(label="Custom"):
                        dpg.add_text("place drones/objects, toggle everything, then press Run",
                                     color=(150, 160, 175))
                        with dpg.collapsing_header(label="Drones"):       # start CLOSED
                            dpg.add_button(label="Add drone (click map)", tag="add_drone_btn",
                                           width=-1, callback=lambda: _set_mode("drone"))
                            dpg.add_slider_float(label="Radius (m)", tag="radius_slider",
                                                 default_value=20.0, min_value=2.0, max_value=50.0)
                            dpg.add_text("drones: 0", tag="drone_count")
                            dpg.add_button(label="Clear drones", width=-1, callback=_on_clear_drones)
                        with dpg.collapsing_header(label="Objects"):
                            with dpg.tree_node(label="Manual", default_open=True):
                                dpg.add_button(label="Add object (click map)", tag="add_object_btn",
                                               width=-1, callback=lambda: _set_mode("object"))
                                dpg.add_slider_float(label="Object speed", tag="speed_slider",
                                                     default_value=6.0, min_value=0.5, max_value=15.0)
                                dpg.add_text("objects: 0", tag="object_count")
                                dpg.add_button(label="Clear objects", width=-1, callback=_on_clear_objects)
                            with dpg.tree_node(label="Preset", default_open=True):
                                dpg.add_combo(list(_PRESET_MAP), default_value="Random walk",
                                              tag="preset_combo", label="Path")
                                dpg.add_slider_int(label="N objects", tag="preset_n_slider",
                                                   default_value=6, min_value=1, max_value=20)
                                dpg.add_slider_float(label="Speed", tag="preset_speed_slider",
                                                     default_value=6.0, min_value=0.5, max_value=15.0)
                                dpg.add_button(label="Deploy", tag="deploy_btn", width=-1, callback=_on_deploy)
                                dpg.add_button(label="Clear preset", width=-1, callback=_on_clear_preset)
                        with dpg.collapsing_header(label="Heat map"):
                            dpg.add_checkbox(label="Coverage shading (drone view)", tag="show_heat_check",
                                             default_value=True, callback=_on_toggle_heat)
                            dpg.add_slider_float(label="Decay rate", tag="decay_slider",
                                                 default_value=state.coverage_decay,
                                                 min_value=0.0, max_value=1.5)
                            dpg.add_slider_float(label="Belief boost", tag="belief_slider",
                                                 default_value=2.2, min_value=0.5, max_value=5.0,
                                                 callback=lambda: draw_scene(STATE))
                        with dpg.collapsing_header(label="Tracking"):
                            dpg.add_combo(["gmphd", "none"], default_value="gmphd", tag="tracker_combo")
                        with dpg.collapsing_header(label="Repositioning", default_open=True):
                            dpg.add_combo(_METHODS, default_value="none", tag="repos_combo",
                                          callback=_on_repos_change)
                            dpg.add_checkbox(label="Show repositioning target", tag="show_repos_check",
                                             default_value=True, callback=lambda: draw_scene(STATE))
                            dpg.add_checkbox(label="Show Voronoi cells (isotropic_voronoi)",
                                             tag="show_voronoi_check", default_value=True,
                                             callback=lambda: draw_scene(STATE))
                            # legend for the repositioning overlays
                            with dpg.group(horizontal=True):
                                dpg.add_text("o", color=C_CENTROID[:3])
                                dpg.add_text("gold dot/line = Voronoi cell centroid the drone "
                                             "steers toward", color=(150, 160, 175))
                            dpg.add_text("Voronoi cells: each colour = one drone's coverage region "
                                         "(only for isotropic_voronoi)", color=(150, 160, 175),
                                         wrap=300)
                        with dpg.collapsing_header(label="Simulation"):
                            dpg.add_slider_int(label="Steps", tag="steps_slider",
                                               default_value=state.n_steps, min_value=10, max_value=300)
                            dpg.add_slider_float(label="dt (s)", tag="dt_slider",
                                                 default_value=state.dt, min_value=0.02, max_value=1.0)
                            dpg.add_slider_int(label="Seed", tag="seed_slider",
                                               default_value=state.seed, min_value=0, max_value=9999)
                    # ---------- SINGLE METHOD TEST ----------
                    with dpg.tab(label="Single"):
                        dpg.add_text("score ONE method over N random epochs", color=(150, 160, 175))
                        dpg.add_combo(_METHODS, default_value="isotropic_voronoi",
                                      tag="single_method", label="Method")
                        dpg.add_combo(["gmphd", "none"], default_value="gmphd",
                                      tag="single_tracker", label="Tracker")
                        dpg.add_combo(list(_PRESET_MAP), default_value="Random walk",
                                      tag="single_motion", label="Motion")
                        dpg.add_slider_int(label="Epochs", tag="single_epochs",
                                           default_value=10, min_value=1, max_value=100)
                        dpg.add_slider_int(label="Random drones", tag="single_drones",
                                           default_value=4, min_value=1, max_value=20,
                                           callback=_on_random_drones_preview)
                        dpg.add_slider_int(label="Random objects", tag="single_objects",
                                           default_value=5, min_value=1, max_value=20)
                        dpg.add_checkbox(label="Watch one episode", tag="single_watch", default_value=False)
                        with dpg.collapsing_header(label="Metrics"):
                            _build_metric_checkboxes("single")
                        dpg.add_button(label="Run", tag="single_run_btn", width=-1, height=30,
                                       callback=_on_run_single)
                    # ---------- MULTI-SCREEN (staged) ----------
                    with dpg.tab(label="Multi"):
                        dpg.add_text("2-4 synced panes: same scenario, methods differ.",
                                     color=(150, 160, 175))
                        dpg.add_text("(split-canvas comparison coming next)", color=(200, 160, 120))
                    # ---------- BATCH EVAL ----------
                    with dpg.tab(label="Batch"):
                        dpg.add_text("headless sweep -> table + CSV (results/Repositioning)",
                                     color=(150, 160, 175))
                        dpg.add_slider_int(label="Epochs/round", tag="batch_epochs",
                                           default_value=10, min_value=1, max_value=100)
                        dpg.add_slider_int(label="Rounds", tag="batch_rounds",
                                           default_value=1, min_value=1, max_value=20)
                        dpg.add_slider_int(label="Random drones", tag="batch_drones",
                                           default_value=4, min_value=1, max_value=20)
                        dpg.add_slider_int(label="Random objects", tag="batch_objects",
                                           default_value=5, min_value=1, max_value=20)
                        dpg.add_combo(list(_PRESET_MAP), default_value="Random walk",
                                      tag="batch_motion", label="Motion")
                        dpg.add_slider_int(label="Parallel (batch size)", tag="batch_size_slider",
                                           default_value=4, min_value=1, max_value=16)
                        dpg.add_text("Methods to compare:")
                        for m, tag, lbl in (("none", "bm_none", "none"),
                                            ("isotropic_voronoi", "bm_voronoi", "isotropic_voronoi"),
                                            ("greedy_mi", "bm_greedy", "greedy_mi"),
                                            ("rsp", "bm_rsp", "rsp"), ("minimax", "bm_minimax", "minimax")):
                            dpg.add_checkbox(label=lbl, tag=tag,
                                             default_value=(m in ("none", "isotropic_voronoi")))
                        with dpg.collapsing_header(label="Metrics"):
                            _build_metric_checkboxes("batch")
                        dpg.add_button(label="Run Batch", tag="batch_run_btn", width=-1, height=30,
                                       callback=_on_run_batch_gui)


def _tick():
    st = STATE
    if st._pending_play:                               # a run worker finished -> start playback here
        st._pending_play = False
        st.frame_idx, st.playing, st._last = 0, True, time.time()
        dpg.configure_item("frame_slider", max_value=max(st.n_steps - 1, 0))
        dpg.set_value("frame_slider", 0)
        _clear_log()
        _set_status(f"ran {st.n_steps} steps - playing")
        draw_scene(st)
        _log_frame(0)
        _update_metrics(0)
        dpg.configure_item("run_btn", enabled=True, label="Run")
        dpg.configure_item("run_progress", show=False)
    if st.playing and st.result is not None:
        now = time.time()
        if now - st._last >= 1.0 / PLAY_FPS:
            st._last = now
            n = len(st.result["frames"])
            if st.frame_idx < n - 1:
                st.frame_idx += 1
                dpg.set_value("frame_slider", st.frame_idx)
                draw_scene(st)
                _log_frame(st.frame_idx)
                _update_metrics(st.frame_idx)
            else:
                st.playing = False
                dpg.set_item_label("play_btn", "Play")
                _set_status("done")


def _warm_up():
    """Compile the JAX paths once in the background so the first Run isn't delayed by JIT."""
    try:
        for method in ("isotropic_voronoi", "greedy_mi"):
            run_placed_tracking([(50.0, 50.0, 18.0)], [(52.0, 52.0)], n_steps=2, dt=0.1,
                                area_xy=(0.0, 100.0, 0.0, 100.0), tracker="gmphd",
                                repositioner=method)
    except Exception:
        pass


def main():
    dpg.create_context()
    _apply_theme()
    build_ui(STATE)
    draw_scene(STATE)
    threading.Thread(target=_warm_up, daemon=True).start()   # precompile JAX off the UI thread
    dpg.create_viewport(title="Drone Tracking Sim", width=1500, height=860)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    while dpg.is_dearpygui_running():
        _tick()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
