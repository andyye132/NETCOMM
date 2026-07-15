"""Run the 2D BEV tracking sim and render a BEV animation.

Drones act as downward-facing cameras, detect ground targets, and a toggleable
tracker (GM-PHD) estimates them. Saves an animated GIF + a final-frame PNG and
prints a short tracking summary.

Examples:
    python -m experiments.run_tracking_demo
    python -m experiments.run_tracking_demo --scenario urban_canyon --steps 60 --targets 5
    python -m experiments.run_tracking_demo --tracker none      # tracking off
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # default to CPU (override by exporting)

import argparse
from pathlib import Path

import numpy as np
import jax

from experiments._runner import load_scenario_method, build_cfg
from netcomm.tracking import run_tracking_episode, CameraSensorConfig, TargetConfig
from netcomm.tracking.visualize import animate_tracking, plot_frame

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Run the BEV tracking sim and render it.")
    ap.add_argument("--scenario", default="open_field",
                    help="config scenario (open_field, urban_canyon, indoor_warehouse, ...)")
    ap.add_argument("--steps", type=int, default=40, help="number of sim steps")
    ap.add_argument("--targets", type=int, default=3, help="number of ground targets")
    ap.add_argument("--tracker", default="gmphd", choices=["none", "gmphd"],
                    help="tracker toggle")
    ap.add_argument("--repositioner", default="none", choices=["none", "isotropic_voronoi"],
                    help="drone repositioning controller (isotropic_voronoi = Cortes coverage control)")
    ap.add_argument("--fov-deg", type=float, default=55.0, help="camera half field-of-view (deg)")
    ap.add_argument("--target-speed", type=float, default=2.0, help="max target speed (m/s)")
    ap.add_argument("--seed", type=int, default=3, help="random seed")
    ap.add_argument("--fps", type=int, default=6, help="animation frames per second")
    ap.add_argument("--out", default=str(REPO / "results" / "tracking"),
                    help="output directory")
    ap.add_argument("--no-gif", action="store_true", help="skip the (slower) GIF render")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scn, _ = load_scenario_method(args.scenario, "adaptive")
    cfg = build_cfg(scn)
    sensor = CameraSensorConfig(half_fov_rad=np.deg2rad(args.fov_deg))
    targets = TargetConfig(n_targets=args.targets, area_xy=cfg.area_xy, v_max=args.target_speed)

    print(f"scenario={args.scenario}  drones={cfg.n_nodes}  area={cfg.area_xy}  "
          f"targets={args.targets}  tracker={args.tracker}  repositioner={args.repositioner}  "
          f"steps={args.steps}")

    result = run_tracking_episode(cfg, n_steps=args.steps, key=jax.random.PRNGKey(args.seed),
                                  tracker=args.tracker, repositioner=args.repositioner,
                                  sensor_cfg=sensor, target_cfg=targets)

    det = [len(f["detections"]) for f in result["frames"]]
    est = [f["n_estimates"] for f in result["frames"]]
    print(f"detections/frame: mean {np.mean(det):.1f} (max {max(det)})   "
          f"estimates/frame: last={est[-1]}")
    if "final_cardinality" in result:
        print(f"final cardinality (expected #targets): {result['final_cardinality']:.2f}")

    last = result["frames"][-1]
    truth, ests = last["targets"], last["estimates"]
    for (pos, P, w) in ests:
        j = int(np.argmin([np.linalg.norm(pos - t) for t in truth]))
        err = float(np.linalg.norm(pos - truth[j]))
        sigma = float(np.sqrt(np.trace(P) / 2))
        print(f"  track {pos.round(1)} w={w:.2f} -> truth {truth[j].round(1)}  "
              f"err={err:.2f}m  uncert~{sigma:.2f}m")

    png = plot_frame(result, str(out_dir / "frame.png"))
    print(f"wrote {png}")
    if not args.no_gif:
        gif = animate_tracking(result, str(out_dir / "tracking.gif"), fps=args.fps)
        print(f"wrote {gif}")


if __name__ == "__main__":
    main()
