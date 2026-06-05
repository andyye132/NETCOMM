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


V_GRID = (1.0, 5.0, 15.0)
N_TRIALS = 100
N_STEPS = 40
METHOD = "adaptive"
TABLE = "calibration"


def main():
    rows = []
    for scn in SCENARIOS:
        for v in V_GRID:
            scn_d, mth_d = load_scenario_method(scn, METHOD)
            cfg = build_cfg(scn_d)
            controller = build_controller(mth_d, cfg)
            flows = default_flows(cfg)
            for trial in range(N_TRIALS):
                key = jax.random.PRNGKey(hash(("cal", scn, v, trial)) & 0xFFFFFFFF)
                t0 = time.perf_counter()
                out = run_episode(cfg, controller, flows, N_STEPS, key)
                wall = time.perf_counter() - t0
                vop_log = out.get("vop_log", []) or []  # includes S_pred per packet
                for entry in vop_log:
                    rows.append(dict(
                        scenario=scn, velocity=v, trial=trial,
                        s_pred=float(entry.get("s_pred", entry.get("S_pred", 0.0))),
                        delivered=int(bool(entry.get("delivered", False))),
                        action=str(entry.get("action", "")),
                        wall=float(wall),
                    ))
            write_parquet(rows, TABLE)
            print(f"[{scn}|v={v}] {N_TRIALS} trials done")
    write_parquet(rows, TABLE)
    print("DONE calibration")


if __name__ == "__main__":
    main()
