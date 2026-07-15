"""Headless batch evaluation: score repositioner methods over Monte-Carlo epochs.

Runs each selected method over `--epochs` randomized episodes (parallelized across cores
via `--batch-size`), prints a mean+/-std table, and writes a CSV to results/Repositioning/.

Examples:
    python -m experiments.run_batch_tests
    python -m experiments.run_batch_tests --methods none isotropic_voronoi greedy_mi rsp \
        --epochs 20 --batch-size 8 --drones 5 --targets 6 --motion circle --name circle_sweep
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

from netcomm.tracking.testing import TestConfig, run_batch, format_table


def main():
    ap = argparse.ArgumentParser(description="Batch-score repositioner methods -> table + CSV.")
    ap.add_argument("--methods", nargs="+",
                    default=["none", "isotropic_voronoi", "greedy_mi", "rsp"],
                    help="repositioner methods to compare "
                         "(available: none isotropic_voronoi greedy_mi rsp minimax)")
    ap.add_argument("--tracker", default="gmphd", choices=["gmphd", "none"])
    ap.add_argument("--epochs", type=int, default=10, help="episodes per method per round")
    ap.add_argument("--rounds", type=int, default=1,
                    help="rounds: each is a fresh-seeded N-epoch batch; methods share seeds within a round")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="parallel episodes (CPU cores used at once)")
    ap.add_argument("--drones", type=int, default=4, help="randomly-placed drones per epoch")
    ap.add_argument("--targets", type=int, default=5)
    ap.add_argument("--motion", default="random_walk",
                    choices=["random_walk", "figure8", "circle", "square", "triangle"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--fov-deg", type=float, default=55.0)
    ap.add_argument("--motion-model", default="cv", choices=["cv", "ca"],
                    help="PCRLB target-state model")
    ap.add_argument("--los-nlos", action="store_true")
    ap.add_argument("--name", default="batch", help="CSV file name (results/Repositioning/<name>.csv)")
    args = ap.parse_args()

    cfg = TestConfig(tracker=args.tracker, n_drones=args.drones, n_targets=args.targets,
                     target_motion=args.motion, n_steps=args.steps, dt=args.dt,
                     object_speed=args.speed, fov_deg=args.fov_deg,
                     los_nlos=args.los_nlos, eval_motion=args.motion_model)
    print(f"batch: methods={args.methods}  epochs={args.epochs}  rounds={args.rounds}  "
          f"batch_size={args.batch_size}  drones={args.drones} targets={args.targets} "
          f"motion={args.motion} steps={args.steps}\n")

    rows, csv_path = run_batch(args.methods, cfg, n_epochs=args.epochs, n_rounds=args.rounds,
                               batch_size=args.batch_size, name=args.name,
                               progress=lambda i, n: print(f"  round {i}/{n} done"))
    print("\n" + format_table(rows))
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
