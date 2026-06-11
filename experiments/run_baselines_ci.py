import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import gc
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


def _drain():
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass


_ALL_METHODS = (
    "gpsr", "glsr", "aodv", "dsr", "olsr", "polsr", "tgpsr",
    "scalar_bfs_predictive", "p3", "car", "learning_router", "gnn_routing",
    "adaptive", "oracle_future",
)
_filter = os.environ.get("METHOD_FILTER", "").strip()
METHODS = tuple(m for m in _ALL_METHODS if m == _filter) if _filter else _ALL_METHODS
N_TRIALS = 50
N_STEPS = 40
TABLE = f"baselines/{_filter}" if _filter else "baselines"


def main():
    rows = []
    for scn in SCENARIOS:
        for mth in METHODS:
            scn_d, mth_d = load_scenario_method(scn, mth)
            cfg = build_cfg(scn_d)
            controller = build_controller(mth_d, cfg)
            flows = default_flows(cfg)
            for trial in range(N_TRIALS):
                key = jax.random.PRNGKey(hash(("bl", scn, mth, trial)) & 0xFFFFFFFF)
                t0 = time.perf_counter()
                out = run_episode(cfg, controller, flows, N_STEPS, key)
                wall = time.perf_counter() - t0
                rows.append(dict(
                    scenario=scn, method=mth, trial=trial,
                    delivery=float(out["delivery_probability"]),
                    mean_aoi=float(out["mean_aoi"]),
                    p99_latency=float(out["p99_latency"]),
                    n_dropped_stale=int(out.get("n_dropped_stale", 0)),
                    control_bytes=int(out.get("control_bytes", 0)),
                    wall=float(wall),
                ))
            write_parquet(rows, TABLE)
            print(f"[{scn}|{mth}] {N_TRIALS} trials done", flush=True)
            _drain()
    write_parquet(rows, TABLE)
    print("DONE baselines_ci")


if __name__ == "__main__":
    main()
