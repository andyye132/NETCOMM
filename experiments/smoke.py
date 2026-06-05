import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import sys
import math
import time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax
import numpy as np

from netcomm.runner import run_episode
from experiments._runner import (
    load_scenario_method, build_cfg, build_controller, default_flows,
)


SCENARIO = "open_field"
METHODS = ("adaptive", "gpsr", "always_predict")
N_TRIALS = 3
N_STEPS = 50


def _is_finite(x):
    try:
        return bool(np.all(np.isfinite(np.asarray(x))))
    except Exception:
        return True


def main():
    print("=== NETCOMM smoke check ===")
    deliveries = {m: [] for m in METHODS}
    runtime_per_pkt = []
    actions_seen = set()
    all_beliefs_finite = True
    n_pkt_total = 0

    for mth in METHODS:
        scn_d, mth_d = load_scenario_method(SCENARIO, mth)
        cfg = build_cfg(scn_d)
        controller = build_controller(mth_d, cfg)
        flows = default_flows(cfg)
        for trial in range(N_TRIALS):
            key = jax.random.PRNGKey(hash(("smoke", mth, trial)) & 0xFFFFFFFF)
            t0 = time.perf_counter()
            out = run_episode(cfg, controller, flows, N_STEPS, key,
                              record_snapshots=True)
            wall = time.perf_counter() - t0
            deliveries[mth].append(float(out["delivery_probability"]))
            n_pkt = int(out.get("n_packets", 1))
            n_pkt_total += n_pkt
            if n_pkt > 0:
                runtime_per_pkt.append(wall / max(n_pkt, 1))
            snaps = out.get("snapshots", []) or []
            for s in snaps:
                rb = s.get("regime_belief", None)
                if rb is not None and not _is_finite(rb):
                    all_beliefs_finite = False
            for entry in out.get("vop_log", []) or []:
                a = entry.get("action", None)
                if a is not None:
                    actions_seen.add(str(a))

    # 1. NaN/Inf in regime_belief
    ok_finite = all_beliefs_finite
    print(f"[1] regime_belief finite: {'PASS' if ok_finite else 'FAIL'}")

    # 2. all four actions
    needed = {"react", "predict", "diversify", "drop"}
    ok_actions = needed.issubset(actions_seen)
    print(f"[2] actions seen {sorted(actions_seen)} -- {'PASS' if ok_actions else 'FAIL (missing ' + str(sorted(needed - actions_seen)) + ')'}")

    # 3. mean delivery in (0,1) for adaptive and gpsr
    md_a = float(np.mean(deliveries["adaptive"])) if deliveries["adaptive"] else float("nan")
    md_g = float(np.mean(deliveries["gpsr"])) if deliveries["gpsr"] else float("nan")
    ok_delivery = (0.0 < md_a < 1.0) and (0.0 < md_g < 1.0)
    print(f"[3] mean delivery adaptive={md_a:.3f} gpsr={md_g:.3f} -- {'PASS' if ok_delivery else 'FAIL'}")

    # 4. per-packet runtime
    avg_rtp = float(np.mean(runtime_per_pkt)) if runtime_per_pkt else float("inf")
    ok_runtime = avg_rtp < 0.050
    print(f"[4] per-packet runtime {avg_rtp*1e3:.2f} ms (N=16) -- {'PASS' if ok_runtime else 'FAIL'}")

    print(f"=== summary: n_packets_total={n_pkt_total}, "
          f"checks={'ALL PASS' if all([ok_finite, ok_actions, ok_delivery, ok_runtime]) else 'FAIL'} ===")


if __name__ == "__main__":
    main()
