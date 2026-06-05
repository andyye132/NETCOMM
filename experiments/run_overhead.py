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


METHODS = ("adaptive", "always_predict", "always_diversify")
N_TRIALS = 30
N_STEPS = 40
TABLE = "overhead"


def main():
    rows = []
    for scn in SCENARIOS:
        for mth in METHODS:
            scn_d, mth_d = load_scenario_method(scn, mth)
            cfg = build_cfg(scn_d)
            controller = build_controller(mth_d, cfg)
            flows = default_flows(cfg)
            for trial in range(N_TRIALS):
                key = jax.random.PRNGKey(hash(("oh", scn, mth, trial)) & 0xFFFFFFFF)
                t0 = time.perf_counter()
                out = run_episode(cfg, controller, flows, N_STEPS, key)
                wall = time.perf_counter() - t0
                n_pkt = max(int(out.get("n_packets", 1)), 1)
                n_delivered = max(int(out.get("delivery_probability", 0.0) * n_pkt), 1)
                rt_step = out.get("runtime_per_step", None)
                mean_rt = float(rt_step[-1]) if rt_step is not None and len(rt_step) else float(wall / N_STEPS)
                rows.append(dict(
                    scenario=scn, method=mth, trial=trial,
                    delivery=float(out["delivery_probability"]),
                    mean_aoi=float(out["mean_aoi"]),
                    p99_latency=float(out["p99_latency"]),
                    n_packets=n_pkt,
                    n_dropped_stale=int(out.get("n_dropped_stale", 0)),
                    compute_per_step=float(mean_rt),
                    compute_per_delivered=float(wall / n_delivered),
                    control_bytes=int(out.get("control_bytes", 0)),
                    control_bytes_per_delivered=float(out.get("control_bytes", 0) / n_delivered),
                    wall=float(wall),
                ))
            write_parquet(rows, TABLE)
            print(f"[{scn}|{mth}] {N_TRIALS} trials done")
    write_parquet(rows, TABLE)
    print("DONE overhead")


if __name__ == "__main__":
    main()
