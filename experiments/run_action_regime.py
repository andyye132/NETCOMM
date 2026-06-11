import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import jax
import pandas as pd  # noqa: F401  (used transitively via write_parquet)
from netcomm.runner import run_episode
from experiments._runner import (
    load_scenario_method, build_cfg, build_controller, default_flows, write_parquet,
    SCENARIOS,
)

METHOD = "adaptive"
N_TRIALS = 10
N_STEPS = 80
TABLE = "action_regime"


def main():
    rows = []
    for scn in SCENARIOS:
        scn_d, mth_d = load_scenario_method(scn, METHOD)
        cfg = build_cfg(scn_d)
        ctrl = build_controller(mth_d, cfg)
        flows = default_flows(cfg)
        for trial in range(N_TRIALS):
            key = jax.random.PRNGKey(hash(("ar", scn, trial)) & 0xFFFFFFFF)
            out = run_episode(cfg, ctrl, flows, N_STEPS, key)
            for entry in out.get("packet_log", []) or []:
                rows.append(dict(
                    scenario=scn, trial=trial,
                    step=int(entry.get("step", -1)),
                    action=str(entry.get("action", "")),
                    regime_argmax=int(entry.get("regime_argmax", -1)),
                    delivered=int(bool(entry.get("delivered", False))),
                    s_pred=float(entry.get("s_pred", 0.0)),
                    vop=float(entry.get("vop", 0.0)),
                    vod=float(entry.get("vod", 0.0)),
                ))
            print(f"[{scn}|t={trial}] packets={len(out.get('packet_log',[]))}", flush=True)
        write_parquet(rows, TABLE)
    write_parquet(rows, TABLE)
    print("DONE action_regime")


if __name__ == "__main__":
    main()
