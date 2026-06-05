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


# label -> (method_yaml, override_dict)
ABLATIONS = {
    "full":                    ("adaptive",     {}),
    "no_hmm":                  ("no_hmm",       {}),
    "no_vop":                  ("no_vop",       {}),
    "no_vod":                  ("no_vod",       {}),
    "no_lcb":                  ("no_lcb",       {}),
    "no_diversify":            ("no_diversify", {}),
    "greedy_diversify":        ("adaptive",     {"diversify_mode": "greedy"}),
    "ot_diversify":            ("adaptive",     {"diversify_mode": "optimal_transport"}),
    "2state_hmm":              ("adaptive",     {"n_regimes": 2}),
}
N_TRIALS = 30
N_STEPS = 40
TABLE = "ablations"


def main():
    rows = []
    for scn in SCENARIOS:
        for label, (mth, ov) in ABLATIONS.items():
            scn_d, mth_d = load_scenario_method(scn, mth)
            cfg = build_cfg(scn_d, overrides=ov or None)
            controller = build_controller(mth_d, cfg)
            flows = default_flows(cfg)
            for trial in range(N_TRIALS):
                key = jax.random.PRNGKey(hash(("abl", scn, label, trial)) & 0xFFFFFFFF)
                t0 = time.perf_counter()
                out = run_episode(cfg, controller, flows, N_STEPS, key)
                wall = time.perf_counter() - t0
                rows.append(dict(
                    scenario=scn, ablation=label, trial=trial,
                    delivery=float(out["delivery_probability"]),
                    mean_aoi=float(out["mean_aoi"]),
                    p99_latency=float(out["p99_latency"]),
                    n_dropped_stale=int(out.get("n_dropped_stale", 0)),
                    wall=float(wall),
                ))
            write_parquet(rows, TABLE)
            print(f"[{scn}|{label}] {N_TRIALS} trials done")
    write_parquet(rows, TABLE)
    print("DONE ablations")


if __name__ == "__main__":
    main()
