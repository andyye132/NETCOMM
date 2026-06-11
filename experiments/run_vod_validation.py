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


V_GRID = (1.0, 3.0, 10.0, 20.0)
DENSITY_GRID = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3)
N_TRIALS = 50
N_STEPS = 40
SCENARIO = "open_field"
METHOD = "adaptive"
TABLE = "vod_validation"


def main():
    rows = []
    for v in V_GRID:
        for dens in DENSITY_GRID:
            scn_d, mth_d = load_scenario_method(SCENARIO, METHOD)
            cfg = build_cfg(scn_d, overrides={"lambda_density": dens})
            controller = build_controller(mth_d, cfg)
            flows = default_flows(cfg)
            for trial in range(N_TRIALS):
                key = jax.random.PRNGKey(hash(("vod", v, dens, trial)) & 0xFFFFFFFF)
                t0 = time.perf_counter()
                out = run_episode(cfg, controller, flows, N_STEPS, key)
                wall = time.perf_counter() - t0
                packet_log = out.get("packet_log", []) or []
                for entry in packet_log:
                    rows.append(dict(
                        velocity=v, density=dens, trial=trial,
                        vod=float(entry.get("vod", 0.0)),
                        delivered=int(bool(entry.get("delivered", False))),
                        action=str(entry.get("action", "")),
                        s_pred=float(entry.get("s_pred", 0.0)),
                        wall=float(wall),
                    ))
            write_parquet(rows, TABLE)
            print(f"[v={v}|dens={dens:.2e}] logged {len(packet_log)} packets/trial")
    write_parquet(rows, TABLE)
    print("DONE vod_validation")


if __name__ == "__main__":
    main()
