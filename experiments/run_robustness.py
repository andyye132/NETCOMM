import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import sys
import time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax

from netcomm.runner import run_episode
from experiments._runner import (
    load_scenario_method, build_cfg, build_controller, default_flows,
    write_parquet, SCENARIOS,
)


PERTURBATIONS = (
    ("nominal",          {}),
    ("alpha_pl+25%",     {"alpha_pl_mult": 1.25}),
    ("alpha_pl-25%",     {"alpha_pl_mult": 0.75}),
    ("m_0+50%",          {"m_0_mult": 1.5}),
    ("m_0-50%",          {"m_0_mult": 0.5}),
    ("cluster_kappa*2",  {"cluster_kappa_mult": 2.0}),
    ("cluster_kappa/2",  {"cluster_kappa_mult": 0.5}),
)
METHODS = ("adaptive", "scalar_bfs_predictive")
N_TRIALS = 30
N_STEPS = 40
TABLE = "robustness"


def _apply_mults(scn_d, mults):
    out = dict(scn_d)
    if "alpha_pl_mult" in mults:
        out["alpha_pl"] = float(scn_d["alpha_pl"]) * mults["alpha_pl_mult"]
    if "m_0_mult" in mults:
        out["m_0"] = float(scn_d["m_0"]) * mults["m_0_mult"]
    if "cluster_kappa_mult" in mults:
        out["cluster_kappa"] = float(scn_d["cluster_kappa"]) * mults["cluster_kappa_mult"]
    return out


def main():
    rows = []
    for scn in SCENARIOS:
        for pert_name, mults in PERTURBATIONS:
            for mth in METHODS:
                scn_d, mth_d = load_scenario_method(scn, mth)
                scn_d = _apply_mults(scn_d, mults)
                cfg = build_cfg(scn_d)
                controller = build_controller(mth_d, cfg)
                flows = default_flows(cfg)
                for trial in range(N_TRIALS):
                    key = jax.random.PRNGKey(hash(("rob", scn, pert_name, mth, trial)) & 0xFFFFFFFF)
                    t0 = time.perf_counter()
                    out = run_episode(cfg, controller, flows, N_STEPS, key)
                    wall = time.perf_counter() - t0
                    rows.append(dict(
                        scenario=scn, perturbation=pert_name, method=mth,
                        trial=trial,
                        delivery=float(out["delivery_probability"]),
                        mean_aoi=float(out["mean_aoi"]),
                        p99_latency=float(out["p99_latency"]),
                        n_dropped_stale=int(out.get("n_dropped_stale", 0)),
                        wall=float(wall),
                    ))
                write_parquet(rows, TABLE)
                print(f"[{scn}|{pert_name}|{mth}] {N_TRIALS} trials done")
    write_parquet(rows, TABLE)
    print("DONE robustness")


if __name__ == "__main__":
    main()
