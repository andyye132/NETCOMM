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


# coh_time (ms) -> cluster_kappa override (T_0 / coh_time_s, base T_0=0.05 s)
COH_TIME_MS = (1.0, 3.0, 10.0, 30.0, 100.0)
N_TRIALS = 30
N_STEPS = 40
METHODS = ("adaptive", "always_react", "always_predict", "always_diversify", "oracle_regime")
TABLE = "regime_sweep"


def main():
    rows = []
    for scn in SCENARIOS:
        for coh_ms in COH_TIME_MS:
            cluster_kappa = 0.05 / (coh_ms * 1e-3)  # base T_0=0.05 s
            for mth in METHODS:
                scn_d, mth_d = load_scenario_method(scn, mth)
                cfg = build_cfg(scn_d, overrides={"cluster_kappa": cluster_kappa})
                controller = build_controller(mth_d, cfg)
                flows = default_flows(cfg)
                for trial in range(N_TRIALS):
                    key = jax.random.PRNGKey(hash((scn, coh_ms, mth, trial)) & 0xFFFFFFFF)
                    t0 = time.perf_counter()
                    out = run_episode(cfg, controller, flows, N_STEPS, key)
                    wall = time.perf_counter() - t0
                    rows.append(dict(
                        scenario=scn, method=mth, coh_time_ms=coh_ms,
                        cluster_kappa=cluster_kappa, trial=trial,
                        delivery=float(out["delivery_probability"]),
                        mean_aoi=float(out["mean_aoi"]),
                        p99_latency=float(out["p99_latency"]),
                        n_dropped_stale=int(out.get("n_dropped_stale", 0)),
                        wall=float(wall),
                    ))
                write_parquet(rows, TABLE)
                print(f"[{scn}|{mth}|coh={coh_ms}ms] {N_TRIALS} trials done")
    write_parquet(rows, TABLE)
    print("DONE regime_sweep")


if __name__ == "__main__":
    main()
