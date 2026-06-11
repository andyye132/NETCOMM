import sys
from pathlib import Path
from datetime import date

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
TAB_DIR = RESULTS / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd


def _read(table: str) -> pd.DataFrame:
    p = RESULTS / table / "sweep.parquet"
    if not p.exists():
        p = p.with_suffix(".csv")
        if p.exists():
            return pd.read_csv(p)
        print(f"[skip] {table}")
        return pd.DataFrame()
    return pd.read_parquet(p)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v).replace("_", r"\_")


def _df_to_tabular(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    align = "l" + "c" * (len(cols) - 1) if cols else "l"
    out = [f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule"]
    out.append(" & ".join(_fmt(c) for c in cols) + " \\\\")
    out.append("\\midrule")
    for _, row in df.iterrows():
        out.append(" & ".join(_fmt(v) for v in row.values) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def _write_tex(df: pd.DataFrame, out: Path, caption: str, label: str):
    body = _df_to_tabular(df)
    txt = (
        "\\begin{table}[h]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\footnotesize\n"
        + body + "\n"
        + "\\end{table}\n"
    )
    out.write_text(txt)
    print(f"[tab] -> {out}")


# ---------------------------------------------------------------------------
# Table 1: baseline summary (static descriptor; built from method registry)
# ---------------------------------------------------------------------------

BASELINE_DESCRIPTORS = [
    # method, information used, behavior class
    ("GPSR",   "Position only",                                "Reactive"),
    ("GLSR",   "Position + load",                              "Reactive"),
    ("AODV",   "Route requests",                               "Reactive"),
    ("DSR",    "Source routes",                                "Reactive"),
    ("OLSR",   "Periodic link state",                          "Proactive"),
    ("P-OLSR", "Link state + prediction",                      "Predictive"),
    ("T-GPSR", "Position + trajectory prediction",             "Predictive"),
    ("Scalar BFS predictive", "Mean link-survival prediction", "Predictive"),
    ("P3",     "Position + context",                           "Predictive"),
    ("CAR",    "Context-aware load + position",                "Reactive"),
    ("Learning router (Q-learning approx)", "Engineered score Dijkstra", "Learned"),
    ("GNN routing", "Topology + graph features",               "Learned"),
    ("NETCOMM (adaptive)", "HMM belief + VoP/VoD + LCB",       "Adaptive"),
    ("Future-channel oracle", "Realized future link states",   "Oracle"),
]


def tab1_baseline_summary():
    df = pd.DataFrame(BASELINE_DESCRIPTORS,
                      columns=["Method", "Information used", "Class"])
    _write_tex(df, TAB_DIR / "tab1_baseline_summary.tex",
               caption="Baseline summary.", label="tab:baseline_summary")


# ---------------------------------------------------------------------------
# Table 2: scenario parameters (static from configs/scenarios/*.yaml)
# ---------------------------------------------------------------------------

import yaml

def tab2_scenario_params():
    rows = []
    scn_dir = REPO / "configs" / "scenarios"
    for p in sorted(scn_dir.glob("*.yaml")):
        d = yaml.safe_load(open(p, "r")) or {}
        ax = d.get("area_xy", [0, 0, 0, 0])
        rows.append(dict(
            Scenario=p.stem,
            Env=d.get("env", ""),
            N=d.get("n_nodes", 0),
            Area=f"{ax[1]-ax[0]:.0f} x {ax[3]-ax[2]:.0f}",
            R_rng=d.get("r_rng", 0.0),
            alpha_pl=d.get("alpha_pl", 0.0),
            m_0=d.get("m_0", 0.0),
            Density=d.get("lambda_density", 0.0),
        ))
    df = pd.DataFrame(rows)
    _write_tex(df, TAB_DIR / "tab2_scenario_params.tex",
               caption="Scenario parameters.", label="tab:scenario_params")


# ---------------------------------------------------------------------------
# Table 3: main performance results
# ---------------------------------------------------------------------------

def tab3_main_performance():
    df = _read("baselines")
    if df.empty:
        print("[tab3] empty baselines; emitting placeholder")
        df = pd.DataFrame(columns=["scenario", "method", "delivery", "mean_aoi",
                                    "p99_latency", "control_bytes"])
    agg = df.groupby(["scenario", "method"], as_index=False).agg(
        delivery=("delivery", "mean"),
        aoi_ms=("mean_aoi", "mean"),
        p99_ms=("p99_latency", "mean"),
        ctrl=("control_bytes", "mean") if "control_bytes" in df else ("delivery", "count"),
    )
    _write_tex(agg, TAB_DIR / "tab3_main_performance.tex",
               caption="Main performance results (mean across seeds).",
               label="tab:main_performance")


# ---------------------------------------------------------------------------
# Table 4: complexity summary
# ---------------------------------------------------------------------------

def tab4_complexity():
    df = _read("scalability")
    base = pd.DataFrame([
        dict(component="HMM belief update", asymptotic="O(|E|)"),
        dict(component="Survival LCB per edge", asymptotic="O(|E|)"),
        dict(component="Dijkstra single path", asymptotic="O(|E| log|V|)"),
        dict(component="k-disjoint paths", asymptotic="O(k |E| log|V|)"),
        dict(component="Per-packet decision", asymptotic="O(|E| + k|E|log|V|)"),
    ])
    if df.empty:
        _write_tex(base, TAB_DIR / "tab4_complexity.tex",
                   caption="Complexity summary (asymptotic only — measured runtimes pending).",
                   label="tab:complexity")
        return
    meas = df.groupby("n_nodes", as_index=False).agg(
        runtime_ms=("runtime_per_step", "mean"),
        delivery=("delivery", "mean"),
    )
    meas["runtime_ms"] = meas["runtime_ms"] * 1e3
    _write_tex(base, TAB_DIR / "tab4_complexity_asym.tex",
               caption="Asymptotic complexity per controller component.",
               label="tab:complexity_asym")
    _write_tex(meas, TAB_DIR / "tab4_complexity_meas.tex",
               caption="Measured runtime per step vs.\\ N.",
               label="tab:complexity_meas")


# ---------------------------------------------------------------------------
# Table 5: ablation results
# ---------------------------------------------------------------------------

def tab5_ablations():
    df = _read("ablations")
    if df.empty:
        print("[tab5] empty ablations; emitting placeholder")
        df = pd.DataFrame(columns=["scenario", "ablation", "delivery", "mean_aoi"])
    agg = df.groupby(["scenario", "ablation"], as_index=False).agg(
        delivery=("delivery", "mean"),
        delivery_sem=("delivery", "sem"),
        aoi_ms=("mean_aoi", "mean"),
    )
    _write_tex(agg, TAB_DIR / "tab5_ablations.tex",
               caption="Ablation results (mean across seeds).",
               label="tab:ablations")


def main():
    for fn in (tab1_baseline_summary, tab2_scenario_params,
               tab3_main_performance, tab4_complexity, tab5_ablations):
        try:
            fn()
        except Exception as e:
            print(f"[error] {fn.__name__}: {e}")
    print("DONE make_tables")


if __name__ == "__main__":
    main()
