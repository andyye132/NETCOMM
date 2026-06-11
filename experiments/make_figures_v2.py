import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIG = RESULTS / "figures" / "v2"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=0.95,
              rc={"figure.dpi": 150, "savefig.dpi": 300,
                  "axes.spines.top": False, "axes.spines.right": False,
                  "axes.linewidth": 0.5,
                  "grid.linewidth": 0.4, "grid.alpha": 0.35})

ACTION_COLORS = {"react": "#1f77b4", "predict": "#2ca02c",
                  "diversify": "#ff7f0e", "drop": "#777777"}
REGIME_COLORS = {"stable": "#2ca02c", "predictable": "#1f77b4",
                  "volatile": "#ff7f0e", "blocked": "#d62728"}

REGIME_NAMES = ["stable", "predictable", "volatile", "blocked"]
ACTION_NAMES = ["react", "predict", "diversify", "drop"]


def _read(name: str) -> pd.DataFrame:
    p = RESULTS / name / "sweep.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def fig_action_by_regime():
    df = _read("action_regime")
    if df.empty or "regime_argmax" not in df.columns or "action" not in df.columns:
        print("[skip] action_by_regime: results/action_regime/sweep.parquet missing")
        return
    df = df[df["regime_argmax"].isin([0, 1, 2, 3]) & df["action"].isin(ACTION_NAMES)]
    if df.empty:
        print("[skip] action_by_regime: no rows after filter")
        return
    counts = (df.groupby(["regime_argmax", "action"]).size()
              .unstack(fill_value=0)
              .reindex(index=[0, 1, 2, 3], columns=ACTION_NAMES, fill_value=0))
    row_sum = counts.sum(axis=1).replace(0, 1)
    cond = counts.div(row_sum, axis=0)
    cond.index = REGIME_NAMES

    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    sns.heatmap(cond, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1,
                cbar_kws=dict(label=r"$P(a\mid r)$", shrink=0.8),
                annot_kws=dict(fontsize=7),
                linewidths=0.4, linecolor="white", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    fig.tight_layout()
    fig.savefig(FIG / "action_by_regime.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote action_by_regime.pdf")


def fig_win_margin():
    df = _read("baselines")
    if df.empty or "method" not in df.columns:
        print("[skip] win_margin: baselines missing")
        return
    means = df.groupby(["scenario", "method"])["delivery"].mean().unstack()
    eps = 1e-3
    rows = []
    for s in means.index:
        row = means.loc[s].dropna()
        if "adaptive" not in row:
            continue
        adaptive = row["adaptive"]
        others = row.drop([c for c in ("adaptive", "oracle_future") if c in row.index])
        if others.empty:
            continue
        best = others.max()
        ratio = adaptive / max(best, eps)
        oracle_ratio = (row["oracle_future"] / max(best, eps)) if "oracle_future" in row else np.nan
        rows.append((s, ratio, oracle_ratio))
    if not rows:
        print("[skip] win_margin: empty rows")
        return
    out = pd.DataFrame(rows, columns=["scenario", "ratio", "oracle_ratio"]).sort_values("ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ys = np.arange(len(out))
    ax.barh(ys, out["ratio"].values, color="#1f77b4", edgecolor="white",
            linewidth=0.5, height=0.65, label="adaptive / best non-oracle")
    for y, r in zip(ys, out["ratio"].values):
        ax.text(r + 0.02, y, f"{r:.2f}", va="center", fontsize=7)
    valid = out["oracle_ratio"].notna()
    ax.scatter(out.loc[valid, "oracle_ratio"].values, ys[valid.values],
               marker="d", color="#888888", s=22, zorder=4,
               label="oracle / best non-oracle")
    ax.axvline(1.0, color="gray", linestyle="--", lw=0.8, alpha=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels(out["scenario"].values, fontsize=7)
    ax.set_xlabel("Ratio (vs best non-oracle baseline)")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "win_margin.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote win_margin.pdf")


def fig_delivery_vs_hardness():
    df = _read("baselines")
    if df.empty:
        print("[skip] delivery_vs_hardness: baselines missing")
        return
    methods = ["adaptive", "scalar_bfs_predictive", "gpsr", "aodv", "oracle_future"]
    sub = df[df["method"].isin(methods)]
    agg = sub.groupby(["scenario", "method"])["delivery"].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    gpsr_mean = agg[agg["method"] == "gpsr"].set_index("scenario")["mean"]
    order = gpsr_mean.sort_values(ascending=False).index.tolist()
    if not order:
        order = sorted(df["scenario"].unique())

    styles = {
        "adaptive": dict(color="#d62728", lw=2.0, ls="-", marker="o"),
        "scalar_bfs_predictive": dict(color="#1f77b4", lw=1.0, ls="-", marker="s"),
        "gpsr": dict(color="#2ca02c", lw=1.0, ls="-", marker="^"),
        "aodv": dict(color="#9467bd", lw=1.0, ls="-", marker="v"),
        "oracle_future": dict(color="gray", lw=1.2, ls="--", marker="x"),
    }

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    xs = np.arange(len(order))
    floor = 1e-3
    for m in methods:
        s = agg[agg["method"] == m].set_index("scenario").reindex(order)
        mean = s["mean"].clip(lower=floor).values
        se = s["se"].fillna(0).values
        st = styles[m]
        ax.plot(xs, mean, label=m, ms=4, **st)
        lo = np.clip(mean - se, floor, None)
        hi = mean + se
        ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.12, linewidth=0)
    ax.set_yscale("log")
    ax.set_ylim(floor, 1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=20, fontsize=7)
    ax.set_xlabel("Scenario (easy -> hard)")
    ax.set_ylabel("Delivery (log)")
    ax.legend(loc="lower left", frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "delivery_vs_hardness.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote delivery_vs_hardness.pdf")


def fig_delivery_distribution():
    df = _read("baselines")
    if df.empty:
        print("[skip] delivery_distribution: baselines missing")
        return
    methods = ["adaptive", "scalar_bfs_predictive", "gpsr", "oracle_future"]
    sub = df[df["method"].isin(methods)].copy()
    scenarios = sorted(sub["scenario"].unique())
    palette = {"adaptive": "#d62728", "scalar_bfs_predictive": "#1f77b4",
               "gpsr": "#2ca02c", "oracle_future": "#777777"}

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(4.0, 5.0),
                             sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scen in zip(axes, scenarios):
        s = sub[sub["scenario"] == scen]
        sns.boxplot(data=s, x="delivery", y="method", order=methods, ax=ax,
                    palette=palette, width=0.55, fliersize=1.2, linewidth=0.6)
        ax.set_ylabel(scen, fontsize=7)
        ax.set_xlabel("")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.tick_params(axis="x", labelsize=6.5)
    axes[-1].set_xlabel("Per-trial delivery")
    fig.tight_layout()
    fig.savefig(FIG / "delivery_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote delivery_distribution.pdf")


def fig_ablation_normalized():
    df = _read("ablations")
    if df.empty:
        print("[skip] ablation_normalized: ablations missing")
        return
    means = df.groupby(["scenario", "ablation"])["delivery"].mean().unstack()
    if "full" not in means.columns:
        print("[skip] ablation_normalized: no 'full' baseline")
        return
    norm = means.div(means["full"].replace(0, np.nan), axis=0)
    variants = ["full"] + sorted([c for c in norm.columns if c != "full"])
    norm = norm[variants]
    scenarios = sorted(norm.index.tolist())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.0, 2.0), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    xs = np.arange(len(variants))
    for ax, scen in zip(axes, scenarios):
        ys = norm.loc[scen].values
        ax.axhline(1.0, color="gray", lw=0.6, ls="--", alpha=0.7)
        ax.plot(xs, ys, "-", color="#888888", lw=0.8, zorder=1)
        colors = ["#d62728" if v == "full" else "#1f77b4" for v in variants]
        sizes = [40 if v == "full" else 18 for v in variants]
        ax.scatter(xs, ys, c=colors, s=sizes, edgecolor="white",
                   linewidth=0.5, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(variants, rotation=60, fontsize=6, ha="right")
        ax.set_title(scen, fontsize=7)
        ax.tick_params(axis="y", labelsize=6.5)
    axes[0].set_ylabel("Delivery / full", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_normalized.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote ablation_normalized.pdf")


def fig_pareto():
    df = _read("baselines")
    if df.empty:
        print("[skip] pareto: baselines missing")
        return
    per_method = df.groupby(["method", "scenario"]).agg(
        delivery=("delivery", "mean"),
        wall=("wall", "mean")).reset_index()
    agg = per_method.groupby("method").agg(
        delivery=("delivery", "mean"),
        wall=("wall", "mean")).reset_index()

    # Pareto: maximize delivery, minimize wall
    pts = agg[["wall", "delivery"]].values
    n = len(agg)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1] and (
                pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1]):
                pareto[i] = False
                break
    agg["pareto"] = pareto

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for _, row in agg.iterrows():
        is_adapt = row["method"] == "adaptive"
        is_par = row["pareto"]
        c = "#d62728" if is_adapt else ("#1f77b4" if is_par else "#aaaaaa")
        s = 50 if is_adapt else (32 if is_par else 18)
        ax.scatter(row["wall"], row["delivery"], c=c, s=s,
                   edgecolor="white", linewidth=0.5, zorder=3)
        if is_adapt or is_par:
            ax.annotate(row["method"], (row["wall"], row["delivery"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    if pareto.any():
        front = agg[pareto].sort_values("wall")
        ax.plot(front["wall"], front["delivery"], "--", color="#1f77b4",
                lw=0.8, alpha=0.6, zorder=1)
    ax.set_xlabel("Mean wall time (s)")
    ax.set_ylabel("Mean delivery")
    fig.tight_layout()
    fig.savefig(FIG / "pareto_delivery_overhead.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote pareto_delivery_overhead.pdf")


def fig_belief_action_events():
    p_belief = RESULTS / "regime_trace" / "trace.parquet"
    p_actions = RESULTS / "regime_trace" / "actions.parquet"
    if not (p_belief.exists() and p_actions.exists()):
        print("[skip] belief_action_events: regime_trace missing")
        return
    bel = pd.read_parquet(p_belief)
    act = pd.read_parquet(p_actions)
    t = bel["t"].values

    fig, (ax_b, ax_a) = plt.subplots(2, 1, figsize=(5.6, 2.8),
                                     gridspec_kw=dict(height_ratios=[3, 1],
                                                      hspace=0.08),
                                     sharex=True)
    ax_b.stackplot(t,
                   bel["p_stable"], bel["p_predictable"],
                   bel["p_volatile"], bel["p_blocked"],
                   colors=[REGIME_COLORS[r] for r in REGIME_NAMES],
                   labels=REGIME_NAMES, alpha=0.85, linewidth=0)
    ax_b.set_ylabel("Regime belief $b_t$")
    ax_b.set_ylim(0, 1)
    ax_b.set_yticks([0, 0.5, 1.0])
    ax_b.legend(loc="upper center", ncol=4, frameon=False, fontsize=7,
                bbox_to_anchor=(0.5, 1.22))

    # Map actions onto belief time axis, then mark NEW-action transitions.
    n_act = len(act)
    if n_act > 0 and len(t) > 0:
        idx = np.linspace(0, n_act, len(t) + 1).astype(int)
        modal = []
        for i in range(len(t)):
            chunk = list(act["action"].iloc[idx[i]:idx[i+1]])
            modal.append(Counter(chunk).most_common(1)[0][0] if chunk else None)
        prev = None
        for i, a in enumerate(modal):
            if a is None or a == prev:
                continue
            ax_a.axvline(t[i], color=ACTION_COLORS.get(a, "gray"),
                         lw=1.2, alpha=0.85)
            prev = a
    ax_a.set_yticks([])
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Time step")
    ax_a.set_ylabel("$\\Delta$ action", fontsize=8)
    handles = [mpatches.Patch(color=c, label=k) for k, c in ACTION_COLORS.items()]
    ax_a.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                fontsize=7, bbox_to_anchor=(0.5, -0.45))
    fig.tight_layout()
    fig.savefig(FIG / "belief_action_events.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote belief_action_events.pdf")


def fig_calibration_ece():
    df = _read("calibration")
    if df.empty or "s_pred" not in df.columns:
        print("[skip] calibration_ece: calibration missing")
        return
    df = df.dropna(subset=["s_pred", "delivered"])
    rows = []
    bins = np.linspace(0, 1, 11)
    for scen, g in df.groupby("scenario"):
        brier = float(np.mean((g["s_pred"].values - g["delivered"].values) ** 2))
        b_idx = np.clip(np.digitize(g["s_pred"].values, bins) - 1, 0, 9)
        n_total = len(g)
        ece = 0.0
        for b in range(10):
            mask = b_idx == b
            n_b = mask.sum()
            if n_b == 0:
                continue
            conf = g["s_pred"].values[mask].mean()
            acc = g["delivered"].values[mask].mean()
            ece += (n_b / n_total) * abs(conf - acc)
        rows.append((scen, brier, ece))
    out = pd.DataFrame(rows, columns=["scenario", "brier", "ece"]).sort_values("brier")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.0, 2.5), sharey=False)
    xs = np.arange(len(out))
    ax1.bar(xs, out["brier"].values, color="#2ca02c", edgecolor="white",
            linewidth=0.5, width=0.65)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(out["scenario"].values, rotation=25, fontsize=7, ha="right")
    ax1.set_ylabel("Brier")
    ax2.bar(xs, out["ece"].values, color="#2ca02c", edgecolor="white",
            linewidth=0.5, width=0.65)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(out["scenario"].values, rotation=25, fontsize=7, ha="right")
    ax2.set_ylabel("ECE")
    fig.tight_layout()
    fig.savefig(FIG / "calibration_ece.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote calibration_ece.pdf")


def main():
    for fn in (fig_action_by_regime, fig_win_margin,
               fig_delivery_vs_hardness, fig_delivery_distribution,
               fig_ablation_normalized, fig_pareto,
               fig_belief_action_events, fig_calibration_ece):
        try:
            fn()
        except Exception as e:
            print(f"[err] {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
