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
    write_parquet,
)


N_NODES_GRID = (8, 16, 32, 64, 128)
N_TRIALS = 10
N_STEPS = 40
SCENARIO = "open_field"
METHOD = "adaptive"
TABLE = "scalability"


def main():
    rows = []
    for N in N_NODES_GRID:
        scn_d, mth_d = load_scenario_method(SCENARIO, METHOD)
        # Keep density ~ constant by scaling area sqrt(N).
        area_side = float((N / float(scn_d["lambda_density"])) ** 0.5)
        ov = {"n_nodes": N, "area_xy": [0.0, area_side, 0.0, area_side]}
        cfg = build_cfg(scn_d, overrides=ov)
        controller = build_controller(mth_d, cfg)
        flows = default_flows(cfg)
        for trial in range(N_TRIALS):
            key = jax.random.PRNGKey(hash(("scale", N, trial)) & 0xFFFFFFFF)
            t0 = time.perf_counter()
            out = run_episode(cfg, controller, flows, N_STEPS, key)
            wall = time.perf_counter() - t0
            rt_step = out.get("runtime_per_step", None) or {}
            mean_rt = float(sum(rt_step.values())) if rt_step else float(wall / N_STEPS)
            rows.append(dict(
                n_nodes=N, trial=trial,
                delivery=float(out["delivery_probability"]),
                mean_aoi=float(out["mean_aoi"]),
                p99_latency=float(out["p99_latency"]),
                n_dropped_stale=int(out.get("n_dropped_stale", 0)),
                runtime_per_step=float(mean_rt),
                wall=float(wall),
            ))
        write_parquet(rows, TABLE)
        print(f"[N={N}] {N_TRIALS} trials done")
    write_parquet(rows, TABLE)
    print("DONE scalability")


if __name__ == "__main__":
    main()
