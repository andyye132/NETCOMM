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


V_GRID = (1.0, 3.0, 5.0, 10.0, 20.0, 30.0)
FC_GRID = (2.4e9, 5.8e9, 24e9, 28e9, 60e9)
N_TRIALS = 20
N_STEPS = 40
SCENARIO = "open_field"
METHOD = "adaptive"
TABLE = "mode_occupancy"


def main():
    rows = []
    for v in V_GRID:
        for fc in FC_GRID:
            scn_d, mth_d = load_scenario_method(SCENARIO, METHOD)
            cfg = build_cfg(scn_d, overrides={"f_c": fc})
            controller = build_controller(mth_d, cfg)
            flows = default_flows(cfg)
            for trial in range(N_TRIALS):
                key = jax.random.PRNGKey(hash(("mode", v, fc, trial)) & 0xFFFFFFFF)
                t0 = time.perf_counter()
                out = run_episode(cfg, controller, flows, N_STEPS, key)
                wall = time.perf_counter() - t0
                occ = out.get("mode_occupancy", {}) or {}
                rows.append(dict(
                    velocity=v, f_c=fc, trial=trial,
                    frac_react=float(occ.get("react", 0.0)),
                    frac_predict=float(occ.get("predict", 0.0)),
                    frac_diversify=float(occ.get("diversify", 0.0)),
                    frac_drop=float(occ.get("drop", 0.0)),
                    delivery=float(out["delivery_probability"]),
                    wall=float(wall),
                ))
            write_parquet(rows, TABLE)
            print(f"[v={v}|fc={fc/1e9:.1f}GHz] {N_TRIALS} trials done")
    write_parquet(rows, TABLE)
    print("DONE mode_occupancy")


if __name__ == "__main__":
    main()
