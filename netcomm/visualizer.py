import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib import cm


_CLASS_LABELS = {0: "ground", 1: "rotary-UAV", 2: "fixed-wing", 3: "handheld"}
_CLASS_MARKERS = {0: "s", 1: "^", 2: ">", 3: "o"}
_CLASS_COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff7f0e", 3: "#9467bd"}
_FLOW_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
# why: per-action route colors; matches paper's color convention.
_ACTION_COLORS = {
    "react": "#1f77b4",      # blue
    "predict": "#2ca02c",    # green
    "diversify": "#ff7f0e",  # orange
    "drop": "#7f7f7f",       # gray
}
_REGIME_LABELS = ["stable", "predictable", "volatile", "blocked"]
_REGIME_COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]


def _draw_network(ax, snap, cfg):
    ax.clear()
    pos = snap["positions"]
    valid = snap["valid"]
    cls = snap["class_id"]
    pi_up = snap["pi_up"]
    routes = snap.get("routes", [])
    actions = snap.get("actions", [])
    flows = snap["flows"]
    valid_idx = np.where(valid)[0]

    cmap = cm.get_cmap("RdYlGn")
    seg_list, seg_colors = [], []
    for i in valid_idx:
        for j in valid_idx:
            if j <= i:
                continue
            p = float(pi_up[i, j])
            if p < 0.05:
                continue
            seg_list.append([(pos[i, 0], pos[i, 1]), (pos[j, 0], pos[j, 1])])
            seg_colors.append(cmap(p))
    if seg_list:
        ax.add_collection(LineCollection(seg_list, colors=seg_colors,
                                         linewidths=0.7, alpha=0.5, zorder=1))

    # why: routes may be either a single path (react/predict) or a list of paths
    # (diversify); we draw each, colored by the matching action label.
    for k, route in enumerate(routes):
        action = actions[k] if k < len(actions) else "react"
        col = _ACTION_COLORS.get(action, _FLOW_COLORS[k % len(_FLOW_COLORS)])
        sub_paths = route if (route and isinstance(route[0], list)) else [route]
        for sub in sub_paths:
            clean = [v for v in (sub or []) if v >= 0]
            if len(clean) < 2:
                continue
            xs = [pos[v, 0] for v in clean]
            ys = [pos[v, 1] for v in clean]
            ax.plot(xs, ys, "-", color=col, lw=2.0, alpha=0.85, zorder=3)

    for c, marker in _CLASS_MARKERS.items():
        mask = valid & (cls == c)
        if not mask.any():
            continue
        ax.scatter(pos[mask, 0], pos[mask, 1], marker=marker,
                   s=80 + 2.0 * pos[mask, 2],
                   facecolor=_CLASS_COLORS[c], edgecolor="black", linewidths=0.6,
                   label=_CLASS_LABELS[c], zorder=4)
    for i in valid_idx:
        ax.annotate(str(int(i)), (pos[i, 0], pos[i, 1]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7,
                    color="black", zorder=5)

    for k, (s, d) in enumerate(flows):
        col = _FLOW_COLORS[k % len(_FLOW_COLORS)]
        ax.scatter(pos[s, 0], pos[s, 1], marker="*", s=240,
                   facecolor="none", edgecolor=col, linewidths=2.0, zorder=6)
        ax.scatter(pos[d, 0], pos[d, 1], marker="X", s=180,
                   facecolor=col, edgecolor="black", linewidths=0.8, zorder=6)

    xmn, xmx, ymn, ymx = cfg.area_xy
    ax.set_xlim(xmn - 5, xmx + 5)
    ax.set_ylim(ymn - 5, ymx + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.85)
    ax.set_title(f"t = {snap['t']:>3d}", fontsize=10)


def _draw_regime_bar(ax, snap):
    # why: replaces the old Wonham bar. We average the (N, N, 4) per-link belief
    # over the active flows' source rows to get a single 4-vector to display.
    ax.clear()
    b = snap.get("regime_belief")
    flows = snap.get("flows", [])
    if b is None or len(flows) == 0:
        avg = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        rows = np.stack([b[int(s)].reshape(-1, 4).mean(axis=0) for (s, _) in flows])
        avg = rows.mean(axis=0)
        avg = avg / max(avg.sum(), 1e-9)
    left = 0.0
    for i in range(4):
        w = float(avg[i])
        ax.barh(0, w, left=left, height=0.6, color=_REGIME_COLORS[i],
                edgecolor="white", linewidth=1.0)
        if w > 0.08:
            ax.text(left + w / 2, 0, f"{_REGIME_LABELS[i]}\n{w:.2f}",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")
        left += w
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("regime belief (avg over active flows)")


def _draw_metrics_strip(ax, snapshots, current_idx):
    ax.clear()
    ts = list(range(current_idx + 1))
    dp = [float(snapshots[t].get("delivery", 0.0)) for t in ts]
    stable = []
    flows0 = snapshots[0].get("flows", [])
    for t in ts:
        b = snapshots[t].get("regime_belief")
        if b is None or len(flows0) == 0:
            stable.append(0.0)
        else:
            rows = np.stack([b[int(s)].reshape(-1, 4).mean(axis=0)
                             for (s, _) in flows0])
            avg = rows.mean(axis=0)
            avg = avg / max(avg.sum(), 1e-9)
            stable.append(float(avg[0]))
    ax.plot(ts, dp, "-o", color=_FLOW_COLORS[0], ms=3, lw=1.4,
            label="per-step delivery prob")
    ax.plot(ts, stable, "-s", color=_REGIME_COLORS[0], ms=3, lw=1.4,
            label="P(stable)")
    ax.set_xlim(0, max(1, len(snapshots) - 1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("step"); ax.set_ylabel("probability")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    ax.grid(alpha=0.3)


def generate_gif(snapshots, cfg, out_path, fps=4, dpi=100):
    if not snapshots:
        print("[netcomm.visualizer] no snapshots; nothing to render.")
        return
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(4, 4, height_ratios=[0.5, 6, 0.5, 2])
    ax_net = fig.add_subplot(gs[1, :])
    ax_reg = fig.add_subplot(gs[2, :])
    ax_ts = fig.add_subplot(gs[3, :])
    fig.suptitle("NETCOMM simulation", fontsize=12, fontweight="bold")

    def animate(i):
        _draw_network(ax_net, snapshots[i], cfg)
        _draw_regime_bar(ax_reg, snapshots[i])
        _draw_metrics_strip(ax_ts, snapshots, i)

    anim = FuncAnimation(fig, animate, frames=len(snapshots),
                         interval=1000 // fps, repeat=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[netcomm.visualizer] saved {out_path} ({len(snapshots)} frames)")
