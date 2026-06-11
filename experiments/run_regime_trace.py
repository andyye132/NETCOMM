import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import sys
import time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax
import numpy as np
import pandas as pd

from netcomm.runner import run_episode
from experiments._runner import (
    load_scenario_method, build_cfg, build_controller, default_flows,
)

OUT = REPO / "results" / "regime_trace" / "trace.parquet"
SCENARIO = "urban_canyon"
METHOD = "adaptive"
N_STEPS = 100
SEED = 7


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scn_d, mth_d = load_scenario_method(SCENARIO, METHOD)
    cfg = build_cfg(scn_d)
    controller = build_controller(mth_d, cfg)
    flows = default_flows(cfg)
    key = jax.random.PRNGKey(SEED)

    out = run_episode(cfg, controller, flows, N_STEPS, key, record_snapshots=True)
    snaps = out.get("snapshots", []) or []
    action_log = out.get("action_log", []) or []
    pkt_log = out.get("packet_log", []) or []

    rows = []
    for t, snap in enumerate(snaps):
        b = np.asarray(snap.get("regime_belief"))  # (MAX_N, MAX_N, 4)
        valid = np.asarray(snap.get("valid", None))
        if valid is None or valid.size == 0:
            valid_idx = np.arange(b.shape[0])
        else:
            valid_idx = np.where(valid)[0]
        # why: aggregate over valid edges only — invalid slots carry init prior or zeros
        # and skew the averaged belief. Mean over the actual node sub-grid.
        if valid_idx.size > 0:
            sub = b[np.ix_(valid_idx, valid_idx)]  # (n, n, 4)
            mean_b = sub.mean(axis=(0, 1))
        else:
            mean_b = np.array([0.25, 0.25, 0.25, 0.25])
        s = float(mean_b.sum())
        if s > 1e-9:
            mean_b = mean_b / s
        rows.append(dict(
            t=t,
            p_stable=float(mean_b[0]),
            p_predictable=float(mean_b[1]),
            p_volatile=float(mean_b[2]),
            p_blocked=float(mean_b[3]),
        ))
    df_belief = pd.DataFrame(rows)
    df_belief.to_parquet(OUT, index=False)

    df_act = pd.DataFrame({"action": action_log})
    df_act.to_parquet(OUT.parent / "actions.parquet", index=False)

    print(f"wrote {OUT} (n_steps={len(df_belief)}, n_actions={len(df_act)})")
    print(f"delivery={out.get('delivery_probability'):.3f}, "
          f"actions seen: {set(action_log)}")


if __name__ == "__main__":
    main()
