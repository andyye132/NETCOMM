import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=0.95,
              rc={"figure.dpi": 150, "savefig.dpi": 300,
                  "axes.spines.top": False, "axes.spines.right": False,
                  "axes.linewidth": 0.5,
                  "grid.linewidth": 0.4, "grid.alpha": 0.35})

ACTION_COLORS = {"react": "#1f77b4", "predict": "#2ca02c",
                  "diversify": "#ff7f0e", "drop": "#777777"}
REGIME_COLORS = {"stable": "#2ca02c", "predictable": "#1f77b4",
                  "volatile": "#ff7f0e", "blocked": "#d62728"}


def _read(table: str) -> pd.DataFrame:
    p = RESULTS / table / "sweep.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def fig_regime_sweep():
    df = _read("regime_sweep")
    if df.empty or "coh_time_ms" not in df.columns:
        return
    methods = ["adaptive", "always_react", "always_predict",
               "always_diversify", "oracle_regime"]
    agg = (df[df["method"].isin(methods)]
           .groupby(["method", "coh_time_ms"])["delivery"]
           .agg(["mean", "std"]).reset_index())
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for m in methods:
        sub = agg[agg["method"] == m].sort_values("coh_time_ms")
        if sub.empty:
            continue
        ax.plot(sub["coh_time_ms"], sub["mean"], "o-", lw=1.4, ms=4, label=m)
        ax.fill_between(sub["coh_time_ms"],
                         sub["mean"] - sub["std"] / np.sqrt(len(sub)),
                         sub["mean"] + sub["std"] / np.sqrt(len(sub)), alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel(r"Coherence time $T_{\rm coh}$ (ms)")
    ax.set_ylabel("Delivery probability")
    ax.legend(loc="best", frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "regime_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote regime_sweep.pdf")


def fig_mode_occupancy():
    df = _read("mode_occupancy")
    if df.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.0), sharey=True)
    for ax, (k, label) in zip(axes,
        [("frac_react", "react"), ("frac_predict", "predict"),
         ("frac_diversify", "diversify"), ("frac_drop", "drop")]):
        piv = df.groupby(["velocity", "f_c"])[k].mean().unstack()
        sns.heatmap(piv, ax=ax, vmin=0, vmax=1, cbar=False,
                     cmap="rocket_r", linewidths=0)
        ax.set_title(label, color=ACTION_COLORS[label])
        ax.set_xlabel(r"$f_c$ (Hz)")
        ax.set_ylabel("velocity (m/s)" if k == "frac_react" else "")
        ax.tick_params(labelsize=6)
        ax.set_xticks([])
    fig.colorbar(axes[-1].collections[0], ax=axes, shrink=0.7,
                  label="action fraction")
    fig.savefig(FIG / "mode_occupancy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote mode_occupancy.pdf")


def fig_vop():
    df = _read("vop_validation")
    if df.empty or "vop" not in df.columns:
        return
    df = df.dropna(subset=["vop", "delivered", "action"])
    if df.empty:
        return
    df["bin"] = pd.qcut(df["vop"], q=10, duplicates="drop")
    bin_mid = df.groupby("bin")["vop"].mean()
    pred_mask = df["action"] == "predict"
    react_mask = df["action"] == "react"
    g_pred = df[pred_mask].groupby("bin")["delivered"].mean()
    g_react = df[react_mask].groupby("bin")["delivered"].mean()
    actual = (g_pred - g_react).reindex(bin_mid.index)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.scatter(bin_mid, actual, s=18, color="#1f77b4", edgecolor="white",
                linewidth=0.5, zorder=3)
    lim = max(abs(float(bin_mid.min())), abs(float(bin_mid.max())),
              abs(float(actual.min() if actual.notna().any() else 0.0)),
              abs(float(actual.max() if actual.notna().any() else 0.0))) * 1.1 + 1e-3
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", lw=0.8, zorder=1)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel("Predicted VoP")
    ax.set_ylabel(r"Realized $\Delta\!\delta_{\rm predict-react}$")
    fig.tight_layout()
    fig.savefig(FIG / "vop_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote vop_scatter.pdf")


def fig_vod():
    df = _read("vod_validation")
    if df.empty or "vod" not in df.columns:
        return
    df = df.dropna(subset=["vod", "delivered", "action"])
    if df.empty:
        return
    df["bin"] = pd.qcut(df["vod"], q=10, duplicates="drop")
    bin_mid = df.groupby("bin")["vod"].mean()
    div_mask = df["action"] == "diversify"
    single_mask = df["action"].isin(["react", "predict"])
    g_div = df[div_mask].groupby("bin")["delivered"].mean()
    g_single = df[single_mask].groupby("bin")["delivered"].mean()
    actual = (g_div - g_single).reindex(bin_mid.index)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.scatter(bin_mid, actual, s=18, color="#ff7f0e", edgecolor="white",
                linewidth=0.5, zorder=3)
    lim = max(abs(float(bin_mid.min())), abs(float(bin_mid.max())),
              abs(float(actual.min() if actual.notna().any() else 0.0)),
              abs(float(actual.max() if actual.notna().any() else 0.0))) * 1.1 + 1e-3
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", lw=0.8, zorder=1)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel("Predicted VoD")
    ax.set_ylabel(r"Realized $\Delta\!\delta_{\rm diversify-single}$")
    fig.tight_layout()
    fig.savefig(FIG / "vod_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote vod_scatter.pdf")


def fig_calibration():
    df = _read("calibration")
    if df.empty or "s_pred" not in df.columns:
        return
    df = df.dropna(subset=["s_pred", "delivered"])
    if df.empty:
        return
    df["bin"] = pd.cut(df["s_pred"], bins=np.linspace(0, 1, 11),
                        include_lowest=True)
    grp = df.groupby("bin").agg(pred=("s_pred", "mean"),
                                  obs=("delivered", "mean"),
                                  n=("delivered", "count")).dropna()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="perfect")
    if not grp.empty:
        ax.scatter(grp["pred"], grp["obs"],
                    s=grp["n"] / max(grp["n"].max(), 1) * 80 + 8,
                    color="#1f77b4", edgecolor="white", linewidth=0.5,
                    zorder=3, label="binned")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"Predicted survival $S_{\rm pred}$")
    ax.set_ylabel("Observed delivery rate")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote calibration.pdf")


def fig_baselines_heatmap():
    df = _read("baselines")
    if df.empty:
        return
    piv = df.groupby(["method", "scenario"])["delivery"].mean().unstack()
    order = piv.mean(axis=1).sort_values(ascending=False).index.tolist()
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                 cbar_kws=dict(label="delivery", shrink=0.8),
                 annot_kws=dict(fontsize=6.5), linewidths=0.4, linecolor="white")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "baselines_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote baselines_heatmap.pdf")


def fig_ablations_heatmap():
    df = _read("ablations")
    if df.empty:
        return
    piv = df.groupby(["ablation", "scenario"])["delivery"].mean().unstack()
    order = ["full"] + sorted([a for a in piv.index if a != "full"])
    piv = piv.loc[[a for a in order if a in piv.index]]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                 cbar_kws=dict(label="delivery", shrink=0.8),
                 annot_kws=dict(fontsize=6.5), linewidths=0.4, linecolor="white")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "ablations_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote ablations_heatmap.pdf")


def fig_scalability():
    df = _read("scalability")
    if df.empty or "n_nodes" not in df.columns:
        return
    agg = df.groupby("n_nodes").agg(
        deliv=("delivery", "mean"),
        deliv_std=("delivery", "std"),
        rt=("runtime_per_step", "mean"),
    ).reset_index()
    fig, ax1 = plt.subplots(figsize=(4.0, 2.7))
    color1 = "#1f77b4"; color2 = "#d62728"
    ax1.errorbar(agg["n_nodes"], agg["deliv"], yerr=agg["deliv_std"] / np.sqrt(10),
                  fmt="o-", color=color1, lw=1.3, ms=4, capsize=2.5,
                  label="delivery")
    ax1.set_xlabel(r"Number of nodes $N$")
    ax1.set_ylabel("Delivery", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.plot(agg["n_nodes"], agg["rt"] * 1000, "s--", color=color2, lw=1.3,
              ms=4, label="step runtime")
    ax2.set_ylabel("Step runtime (ms)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / "scalability.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote scalability.pdf")


def fig_regime_switching():
    p_belief = RESULTS / "regime_trace" / "trace.parquet"
    p_actions = RESULTS / "regime_trace" / "actions.parquet"
    if not (p_belief.exists() and p_actions.exists()):
        return
    bel = pd.read_parquet(p_belief)
    act = pd.read_parquet(p_actions)

    fig, (ax_b, ax_a) = plt.subplots(2, 1, figsize=(5.6, 3.0),
                                       gridspec_kw=dict(height_ratios=[3, 1],
                                                          hspace=0.05),
                                       sharex=True)
    t = bel["t"].values
    ax_b.stackplot(t,
                    bel["p_stable"], bel["p_predictable"],
                    bel["p_volatile"], bel["p_blocked"],
                    colors=[REGIME_COLORS["stable"], REGIME_COLORS["predictable"],
                             REGIME_COLORS["volatile"], REGIME_COLORS["blocked"]],
                    labels=["stable", "predictable", "volatile", "blocked"],
                    alpha=0.85, linewidth=0)
    ax_b.set_ylabel("Regime belief $b_t$")
    ax_b.set_ylim(0, 1)
    ax_b.set_yticks([0, 0.5, 1.0])
    ax_b.legend(loc="upper center", ncol=4, frameon=False, fontsize=7,
                 bbox_to_anchor=(0.5, 1.18))

    n_act = len(act)
    if n_act > 0 and len(t) > 0:
        idx = np.linspace(0, n_act, len(t) + 1).astype(int)
        from collections import Counter
        modal = []
        for i in range(len(t)):
            chunk = list(act["action"].iloc[idx[i]:idx[i+1]])
            modal.append(Counter(chunk).most_common(1)[0][0] if chunk else "react")
        for i, a in enumerate(modal):
            ax_a.barh(0, 1, left=t[i], color=ACTION_COLORS.get(a, "gray"),
                       height=0.8, edgecolor="none")
    ax_a.set_yticks([])
    ax_a.set_ylim(-0.4, 0.4)
    ax_a.set_xlabel("Time step")
    ax_a.set_ylabel("Action", fontsize=8)
    handles = [mpatches.Patch(color=c, label=k) for k, c in ACTION_COLORS.items()]
    ax_a.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                 fontsize=7, bbox_to_anchor=(0.5, -0.5))
    fig.tight_layout()
    fig.savefig(FIG / "regime_switching.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote regime_switching.pdf")


def main():
    fig_regime_sweep()
    fig_mode_occupancy()
    fig_vop()
    fig_vod()
    fig_calibration()
    fig_baselines_heatmap()
    fig_ablations_heatmap()
    fig_scalability()
    fig_regime_switching()


if __name__ == "__main__":
    main()
