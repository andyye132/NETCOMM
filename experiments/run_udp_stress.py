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


METHODS = ("adaptive", "gpsr", "aodv")
# bursty rates: 3x baseline for video + urgent
STRESS_RATES = (10.0, 90.0, 15.0)
N_TRIALS = 30
N_STEPS = 40
TABLE = "udp_stress"


def main():
    rows = []
    for scn in SCENARIOS:
        for mth in METHODS:
            scn_d, mth_d = load_scenario_method(scn, mth)
            cfg = build_cfg(scn_d, overrides={"pkt_rate_per_class": list(STRESS_RATES)})
            controller = build_controller(mth_d, cfg)
            flows = default_flows(cfg)
            for trial in range(N_TRIALS):
                key = jax.random.PRNGKey(hash(("udp", scn, mth, trial)) & 0xFFFFFFFF)
                t0 = time.perf_counter()
                out = run_episode(cfg, controller, flows, N_STEPS, key)
                wall = time.perf_counter() - t0
                # Per-priority delivery if available
                pri_d = out.get("delivery_by_priority", [None, None, None])
                rows.append(dict(
                    scenario=scn, method=mth, trial=trial,
                    delivery=float(out["delivery_probability"]),
                    mean_aoi=float(out["mean_aoi"]),
                    p99_latency=float(out["p99_latency"]),
                    n_dropped_stale=int(out.get("n_dropped_stale", 0)),
                    delivery_low=float(pri_d[0] if pri_d[0] is not None else 0.0),
                    delivery_video=float(pri_d[1] if pri_d[1] is not None else 0.0),
                    delivery_urgent=float(pri_d[2] if pri_d[2] is not None else 0.0),
                    wall=float(wall),
                ))
            write_parquet(rows, TABLE)
            print(f"[{scn}|{mth}] {N_TRIALS} trials done")
    write_parquet(rows, TABLE)
    print("DONE udp_stress")


if __name__ == "__main__":
    main()
