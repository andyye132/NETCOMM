import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper")
PALETTE = "Set2"


def _read(table: str) -> pd.DataFrame:
    p = RESULTS / table / "sweep.parquet"
    if not p.exists():
        p = p.with_suffix(".csv")
        if p.exists():
            return pd.read_csv(p)
        print(f"[skip] {table}: no parquet at {p}")
        return pd.DataFrame()
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# Figures 1-4: TikZ / schematic / single-trace -- defer to Agent 1
# ---------------------------------------------------------------------------

def fig1_problem_illustration():
    # TODO: TikZ figure produced separately; nothing to render here.
    pass


def fig2_architecture():
    # TODO: TikZ block diagram produced separately.
    pass


def fig3_belief_over_time():
    # TODO Agent 1 will produce snapshot for Fig 3 (single representative trace).
    pass


def fig4_decision_boundary():
    # TODO: decision-boundary heatmap based on a controller-level grid sweep;
    # deferred to a later Agent 1 pass.
    pass


# ---------------------------------------------------------------------------
# Fig 5: delivery vs coherence time
# ---------------------------------------------------------------------------

def fig5_delivery_vs_coh():
    df = _read("regime_sweep")
    if df.empty:
        return
    agg = df.groupby(["method", "coh_time_ms"], as_index=False).agg(
        delivery_mean=("delivery", "mean"),
        delivery_se=("delivery", "sem"),
    )
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for mth, sub in agg.groupby("method"):
        ax.errorbar(sub["coh_time_ms"], sub["delivery_mean"],
                    yerr=sub["delivery_se"], marker="o", capsize=2, label=mth)
    ax.set_xscale("log")
    ax.set_xlabel("Coherence time (ms)")
    ax.set_ylabel("On-time delivery")
    ax.set_title("Delivery vs. coherence time")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig5_delivery_vs_coh_time.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig5] -> {out}")


# ---------------------------------------------------------------------------
# Fig 6: mode occupancy heatmap (e.g. predict fraction)
# ---------------------------------------------------------------------------

def fig6_mode_occupancy():
    df = _read("mode_occupancy")
    if df.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0), sharey=True)
    for ax, mode, col in zip(
        axes,
        ("react", "predict", "diversify", "drop"),
        ("frac_react", "frac_predict", "frac_diversify", "frac_drop"),
    ):
        pivot = df.pivot_table(index="velocity", columns="f_c",
                                values=col, aggfunc="mean")
        sns.heatmap(pivot, ax=ax, cmap="viridis", vmin=0, vmax=1,
                    cbar_kws={"label": mode})
        ax.set_title(mode)
        ax.set_xlabel("f_c (Hz)")
        ax.set_ylabel("velocity (m/s)")
    fig.tight_layout()
    out = FIG_DIR / "fig6_mode_occupancy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig6] -> {out}")


# ---------------------------------------------------------------------------
# Fig 7: VoP scatter
# ---------------------------------------------------------------------------

def fig7_vop_scatter():
    df = _read("vop_validation")
    if df.empty:
        return
    df["actual_gain"] = df["deliv_predict"] - df["deliv_react"]
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    sns.scatterplot(data=df, x="vop", y="actual_gain", hue="velocity",
                    palette=PALETTE, alpha=0.4, s=10, ax=ax, legend=False)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Predicted VoP")
    ax.set_ylabel("delivery_predict - delivery_react")
    ax.set_title("Value of Prediction")
    fig.tight_layout()
    out = FIG_DIR / "fig7_vop_scatter.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig7] -> {out}")


# ---------------------------------------------------------------------------
# Fig 8: VoD scatter
# ---------------------------------------------------------------------------

def fig8_vod_scatter():
    df = _read("vod_validation")
    if df.empty:
        return
    df["actual_gain"] = df["deliv_diversify"] - df["deliv_best_single"]
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    sns.scatterplot(data=df, x="vod", y="actual_gain", hue="density",
                    palette=PALETTE, alpha=0.4, s=10, ax=ax, legend=False)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Predicted VoD")
    ax.set_ylabel("delivery_diversify - max(react, predict)")
    ax.set_title("Value of Diversification")
    fig.tight_layout()
    out = FIG_DIR / "fig8_vod_scatter.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig8] -> {out}")


# ---------------------------------------------------------------------------
# Fig 9: reliability diagram
# ---------------------------------------------------------------------------

def fig9_calibration():
    df = _read("calibration")
    if df.empty:
        return
    bins = np.linspace(0.0, 1.0, 11)
    df["bin"] = pd.cut(df["s_pred"], bins=bins, include_lowest=True)
    agg = df.groupby("bin", observed=True).agg(
        pred_mean=("s_pred", "mean"),
        obs_freq=("delivered", "mean"),
        n=("delivered", "count"),
    ).reset_index().dropna()
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, label="ideal")
    ax.plot(agg["pred_mean"], agg["obs_freq"], "o-", color="tab:blue",
            label="empirical")
    ax.set_xlabel("Predicted survival")
    ax.set_ylabel("Observed delivery frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "fig9_calibration.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig9] -> {out}")


# ---------------------------------------------------------------------------
# Fig 10: baseline bar chart
# ---------------------------------------------------------------------------

def fig10_baselines():
    df = _read("baselines")
    if df.empty:
        return
    agg = df.groupby(["scenario", "method"], as_index=False).agg(
        delivery_mean=("delivery", "mean"),
        delivery_se=("delivery", "sem"),
    )
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    sns.barplot(data=agg, x="method", y="delivery_mean", hue="scenario",
                palette=PALETTE, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("On-time delivery")
    ax.set_title("Baseline comparison")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    out = FIG_DIR / "fig10_baselines.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig10] -> {out}")


# ---------------------------------------------------------------------------
# Fig 11: ablation bar chart
# ---------------------------------------------------------------------------

def fig11_ablations():
    df = _read("ablations")
    if df.empty:
        return
    agg = df.groupby(["scenario", "ablation"], as_index=False).agg(
        delivery_mean=("delivery", "mean"),
        delivery_se=("delivery", "sem"),
    )
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    sns.barplot(data=agg, x="ablation", y="delivery_mean", hue="scenario",
                palette=PALETTE, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("On-time delivery")
    ax.set_title("Component ablations")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    out = FIG_DIR / "fig11_ablations.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig11] -> {out}")


# ---------------------------------------------------------------------------
# Fig 12: runtime scaling
# ---------------------------------------------------------------------------

def fig12_scalability():
    df = _read("scalability")
    if df.empty:
        return
    agg = df.groupby("n_nodes", as_index=False).agg(
        runtime_mean=("runtime_per_step", "mean"),
        runtime_se=("runtime_per_step", "sem"),
        delivery_mean=("delivery", "mean"),
    )
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.errorbar(agg["n_nodes"], agg["runtime_mean"] * 1e3,
                yerr=agg["runtime_se"] * 1e3, marker="o", capsize=2, color="tab:blue")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes N")
    ax.set_ylabel("Runtime per step (ms)")
    ax.set_title("Runtime scaling")
    fig.tight_layout()
    out = FIG_DIR / "fig12_scalability.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig12] -> {out}")


# ---------------------------------------------------------------------------

ALL_FIGS = (
    fig1_problem_illustration,
    fig2_architecture,
    fig3_belief_over_time,
    fig4_decision_boundary,
    fig5_delivery_vs_coh,
    fig6_mode_occupancy,
    fig7_vop_scatter,
    fig8_vod_scatter,
    fig9_calibration,
    fig10_baselines,
    fig11_ablations,
    fig12_scalability,
)


def main():
    for fn in ALL_FIGS:
        try:
            fn()
        except Exception as e:
            print(f"[error] {fn.__name__}: {e}")
    print("DONE make_figures")


if __name__ == "__main__":
    main()
