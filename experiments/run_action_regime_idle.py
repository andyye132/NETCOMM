import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import sys
import time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax
import pandas as pd

from netcomm.runner import run_episode
from experiments._runner import (
    load_scenario_method, build_cfg, build_controller, default_flows,
)


SCENARIO = "idle_team"
METHOD = "adaptive"
V_MAX = 0.03
N_TRIALS = 10
N_STEPS = 80
OUT_DIR = REPO / "results" / "action_regime"
APPEND_TO = OUT_DIR / "sweep.parquet"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scn_d, mth_d = load_scenario_method(SCENARIO, METHOD)
    cfg = build_cfg(scn_d)
    ctrl = build_controller(mth_d, cfg)
    flows = default_flows(cfg)

    rows = []
    for trial in range(N_TRIALS):
        key = jax.random.PRNGKey(hash(("aridle", trial)) & 0xFFFFFFFF)
        t0 = time.perf_counter()
        out = run_episode(cfg, ctrl, flows, N_STEPS, key, v_max=V_MAX)
        wall = time.perf_counter() - t0
        for entry in out.get("packet_log", []) or []:
            rows.append(dict(
                scenario=SCENARIO,
                trial=trial,
                step=int(entry.get("step", -1)),
                action=str(entry.get("action", "")),
                regime_argmax=int(entry.get("regime_argmax", -1)),
                delivered=int(bool(entry.get("delivered", False))),
                s_pred=float(entry.get("s_pred", 0.0)),
                vop=float(entry.get("vop", 0.0)),
                vod=float(entry.get("vod", 0.0)),
            ))
        print(f"[{SCENARIO}|t={trial}] packets={len(out.get('packet_log',[]))} wall={wall:.1f}s",
              flush=True)

    new = pd.DataFrame(rows)
    if APPEND_TO.exists():
        prev = pd.read_parquet(APPEND_TO)
        prev = prev[prev["scenario"] != SCENARIO]  # replace any old idle_team rows
        out_df = pd.concat([prev, new], ignore_index=True)
    else:
        out_df = new
    out_df.to_parquet(APPEND_TO, index=False)
    print(f"wrote {APPEND_TO}: {out_df.shape}")
    print(f"regimes seen for idle_team: {sorted(new['regime_argmax'].unique().tolist())}")


if __name__ == "__main__":
    main()
