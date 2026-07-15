# NETCOMM Drone-Tracking Sim

A **JAX-based simulator for multi-drone, multi-object tracking**. Drones carry
downward-facing cameras; a **GM-PHD filter** estimates the (unknown number of)
moving targets from their noisy detections, and a pluggable **repositioner**
moves the drones to improve tracking. Because the sim owns the **true target
states**, it benchmarks both the tracker and the repositioner against ground
truth (PCRLB bound, GOSPA, OSPA², MOT/HOTA).

Two roles toggle independently:

- **Tracker** — estimates target states from drone observations. Default
  **GM-PHD** (Vo & Ma 2006).
- **Repositioner** — moves drones to improve tracking. The tracker's estimates
  define an importance density **φ**, and the repositioner optimizes drone
  placement against φ.

## Repositioners

| Toggle | Method | Source |
|---|---|---|
| `none` | Static drones (baseline) | — |
| `isotropic_voronoi` | Centroidal Voronoi / Lloyd coverage | Cortés 2004 |
| `greedy_mi` | Sequential-greedy mutual information | Corah & Michael 2021 |
| `rsp` | Randomized Sequential Partitions + greedy | Corah & Michael 2021 |
| `minimax` | Non-myopic minimax `min_u max_z tr(Σ_T)` | Zhang & Tokekar 2016 |

Each method lives in its own standalone package (`coverage_control/`,
`infomax/`, `nonmyopic/`, `gmphd/`) with paper-faithful tests, plus a thin
adapter in `netcomm/tracking/repositioner.py`. New methods plug in via
`make_repositioner(name, cfg, sensor_cfg)`.

## Layout

| Path | Role |
|---|---|
| `netcomm/tracking/` | Sim core: runner, GUI app, sensors, coverage field, repositioner/tracker adapters, test harness |
| `gmphd/` | GM-PHD filter (tracker) |
| `coverage_control/` | Isotropic Voronoi coverage (Cortés 2004) |
| `infomax/` | Greedy / RSP mutual-information planner (Corah & Michael 2021) |
| `nonmyopic/` | Non-myopic minimax target tracking (Zhang & Tokekar 2016) |
| `evaluation/` | Ground-truth scorecard: PCRLB, GOSPA, OSPA², MOTA/MOTP/IDF1, HOTA |
| `experiments/` | Headless CLI entry points (batch tests, evaluation, demo) |
| `modtrack/` | Alternative tracker — reserved, not yet wired in |

### Legacy base (not part of the tracking sim)

This repo was forked from the upstream Tejadi/NETCOMM **packet-routing**
project. The inherited `netcomm/{routing,channel,regime,world,controller,
diversify,packets,...}` packages and the `netcomm/visualizer.py`,
`netcomm/runner.py` (plus the routing experiment drivers in `experiments/`,
`configs/`, `ns3_validation/`) are **legacy from that upstream project and are
NOT used by the tracking sim**. The tracking sim reuses only **`NetCommWorld`**
(from `netcomm.runner`) for drone placement and linear kinematics; everything
else under `netcomm/tracking/` is new.

## Install

Uses [`uv`](https://docs.astral.sh/uv/). JAX (CPU or CUDA) is required.

```bash
uv sync
```

## Run

```bash
# GUI (dearpygui): build a scenario, run/play/scrub, toggle tracker & repositioner
python -m netcomm.tracking.app

# Headless batch — scores methods against ground truth, writes table + CSV to results/Repositioning/
python experiments/run_batch_tests.py \
    --methods none isotropic_voronoi greedy_mi rsp minimax \
    --epochs 20 --rounds 3 --batch-size 8 \
    --drones 5 --targets 6 --motion random_walk --name sweep
```

Batch flags include `--motion-model {cv,ca}`, `--los-nlos`, `--steps`, `--dt`,
`--speed`, and `--fov-deg`. Within a round every method runs the *identical*
scenarios (matched paired seeds) for a fair comparison.

## Tests

The suite runs under JAX on CPU. `pytest` is declared in the `dev` dependency
group:

```bash
JAX_PLATFORMS=cpu uv run --group dev python -m pytest -q
```

Per-package paper-faithful tests live in each package's `tests/` directory
(`gmphd/tests`, `coverage_control/tests`, `infomax/tests`, `nonmyopic/tests`,
`netcomm/tracking/tests`, `evaluation/tests`).
