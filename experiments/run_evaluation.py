"""Benchmark (tracker, repositioner) pairs against the ground-truth metrics.

Runs the same moving-target scenario under each repositioner and prints the
ground-truth scorecard: GOSPA (accuracy), the PCRLB position bound (best obtainable
error), achieved RMSE, efficiency (achieved/bound), per-step information gain, and
OSPA^2 (track-level). The sim knows the true target states, so these are an
objective answer key for how good each pair is.

Examples:
    python -m experiments.run_evaluation
    python -m experiments.run_evaluation --preset figure8 --targets 6 --los-nlos
    python -m experiments.run_evaluation --motion ca --steps 80
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

import numpy as np

from netcomm.tracking import run_preset_tracking, CameraSensorConfig
from coverage_control import CoverageConfig
from evaluation import evaluate, EvalConfig


def _auto_drones(area, n, radius):
    """Place n drones on a ring matching the preset target orbit, so the stationary
    'none' baseline catches targets as they pass and the repositioner can improve on it."""
    xmn, xmx, ymn, ymx = area
    cx, cy = (xmn + xmx) / 2.0, (ymn + ymx) / 2.0
    r = 0.34 * min(xmx - xmn, ymx - ymn)          # ~ the preset path radius
    return [(cx + r * np.cos(2 * np.pi * i / n), cy + r * np.sin(2 * np.pi * i / n), radius)
            for i in range(n)]


def _repos_cfg(method: str, v_max: float):
    """The method's own config type, with the SHARED --vmax speed cap applied, so every
    method competes under the identical actuation limit."""
    key = (method or "none").lower()
    if key in ("greedy_mi", "rsp"):
        from infomax import InfomaxConfig
        return InfomaxConfig(method="greedy" if key == "greedy_mi" else "rsp", v_max=v_max)
    if key == "minimax":
        from nonmyopic import MinimaxConfig
        return MinimaxConfig(v_max=v_max)
    if key in ("isotropic_voronoi", "voronoi", "lloyd"):
        return CoverageConfig(v_max=v_max)
    return None


def main():
    ap = argparse.ArgumentParser(description="Ground-truth benchmark of tracker/repositioner pairs.")
    ap.add_argument("--preset", default="circle",
                    choices=["figure8", "circle", "square", "triangle"],
                    help="target path family (constant cardinality, required for evaluation)")
    ap.add_argument("--methods", nargs="+", default=["none", "isotropic_voronoi"],
                    help="repositioners to compare: none isotropic_voronoi greedy_mi rsp minimax")
    ap.add_argument("--tracker", default="gmphd", choices=["gmphd", "none"])
    ap.add_argument("--targets", type=int, default=5)
    ap.add_argument("--drones", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--dt", type=float, default=0.2)
    # NB: defaults match experiments/run_batch_tests.py + the method packages' own
    # v_max=10 — the two CLIs must score the same physics or their numbers diverge.
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--vmax", type=float, default=10.0, help="repositioner max drone speed")
    ap.add_argument("--motion", default="cv", choices=["cv", "ca"], help="PCRLB target-state model")
    ap.add_argument("--gospa-c", type=float, default=10.0)
    ap.add_argument("--fov-deg", type=float, default=55.0)
    ap.add_argument("--los-nlos", action="store_true", help="enable LoS/NLoS elevation sensing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    area = (0.0, 300.0, 0.0, 300.0)
    sensor = CameraSensorConfig(half_fov_rad=np.deg2rad(args.fov_deg), los_nlos=args.los_nlos)
    drones = _auto_drones(area, args.drones, radius=22.0)
    eval_cfg = EvalConfig(motion=args.motion, gospa_c=args.gospa_c)

    print(f"preset={args.preset} methods={args.methods} tracker={args.tracker} "
          f"targets={args.targets} drones={args.drones} steps={args.steps} "
          f"dt={args.dt} motion={args.motion} los_nlos={args.los_nlos}\n")

    rows = []
    for repo in args.methods:
        res = run_preset_tracking(
            drones, args.preset, n_objects=args.targets, n_steps=args.steps, dt=args.dt,
            area_xy=area, sensor_cfg=sensor, object_speed=args.speed, tracker=args.tracker,
            repositioner=repo, repos_cfg=_repos_cfg(repo, args.vmax), seed=args.seed)
        # evaluate() reads the dt + sensor_cfg recorded in the result -> guaranteed to
        # score the same physics the run was generated with.
        s = evaluate(res, cfg=eval_cfg)["summary"]
        rows.append((repo, s))

    def _fmt(v, spec):
        return ("{:>" + spec + "}").format(v) if v is not None else f"{'--':>{spec.split('.')[0]}}"

    hdr = (f"{'repositioner':<18}{'GOSPA':>8}{'track%':>7}{'bnd_obs':>8}{'effic':>7}"
           f"{'infogain':>9}{'OSPA2':>7}{'MOTA':>7}{'IDF1':>7}{'HOTA':>7}")
    print("--- RFS / sensor-mgmt headline metrics ----------------- | --- CV translation ---")
    print(hdr)
    print("-" * len(hdr))
    for repo, s in rows:
        print(f"{repo:<18}{s['gospa_mean']:>8.2f}{100 * s['tracked_fraction']:>6.0f}%"
              f"{s['bound_rmse_observed']:>8.2f}{s['efficiency_mean']:>7.2f}"
              f"{s['info_gain_cumulative']:>9.1f}{s['ospa2_mean']:>7.2f}"
              f"{_fmt(s.get('mota'), '7.2f')}{_fmt(s.get('idf1'), '7.2f')}{_fmt(s.get('hota'), '7.2f')}")

    print("\nheadline (RFS): lower GOSPA/bnd_obs/OSPA2, higher track%/infogain; efficiency=achieved/bound (→1).  "
          "\ntranslation (CV): higher MOTA/IDF1/HOTA is better (computed on stitched tracks).")


if __name__ == "__main__":
    main()
