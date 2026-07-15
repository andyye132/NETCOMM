# Repository Overview — what's in this folder

A map of every top-level item. The repo has two layers: the **active tracking sim** (new work)
and an **inherited legacy base** (the upstream Tejadi/NETCOMM packet-routing project, largely
unused by the tracking sim). This document flags which is which.

> Orientation: **the tracking sim** = `netcomm/tracking/` + `gmphd/` `coverage_control/`
> `infomax/` `nonmyopic/` `evaluation/` `diffsim/`. Everything else under `netcomm/` is legacy.
> See `PROJECT_STATE.md` (design) and `DEMO_NOTES.md` (how to run / explain).

---

## Top-level files
| Item | What it is |
|---|---|
| `README.md` | Project intro for the tracking sim. |
| `PROJECT_STATE.md` | Living design doc: purpose, methods, evaluation, status. |
| `DEMO_NOTES.md` | Base & demo notes — how to run and explain each method. |
| `REPO_OVERVIEW.md` | This file — the folder map. |
| `pyproject.toml` / `uv.lock` | Package + pinned deps (managed with `uv`; incl. `dearpygui`, `pytest`). |
| `netcomm.egg-info/` | Auto-generated packaging metadata (ignore). |
| `untitled folder/` | Empty stray dir — safe to delete. |

---

## Active tracking-sim packages (the real work)

### Trackers — "where are the targets?"
| Dir | Role |
|---|---|
| **`gmphd/`** | **GM-PHD filter (default tracker)**, Vo & Ma 2006. `gmphd.py` (recursion), `kernels.py` (JAX jit/vmap kernels), `models.py` (CV + density), `config.py`, `types.py`. |
| `modtrack/` | Cross-sensor fusion **primitives only** (clustering/fusion/linalg/motion/uncertainty). **Not a usable tracker, not wired in** — ignore. |

### Repositioners — "where should the drones go?"
| Dir | Role |
|---|---|
| **`coverage_control/`** | **Isotropic Voronoi / Lloyd coverage** (Cortés 2004). `voronoi.py` (JAX kernels), `controller.py`, `config.py`. Toggle `isotropic_voronoi`. |
| **`infomax/`** | **Greedy / RSP mutual-information** planner (Corah & Michael 2021). `objective.py` (JAX MI), `maximizer.py` (greedy/RSP), `actions.py`, `controller.py`. Toggles `greedy_mi`, `rsp`. |
| **`nonmyopic/`** | **Non-myopic minimax** target tracking (Zhang & Tokekar 2016). `tree.py` (minimax + JAX), `riccati.py` (Eq-7 Kalman/Riccati), `pruning.py`, `assignment.py` (multi-robot greedy), `sampler.py`. Toggle `minimax`. |

### Sim core, evaluation, differentiable core
| Dir | Role |
|---|---|
| **`netcomm/tracking/`** | **The simulator itself.** `runner.py` (episode loops), `sensors.py` (camera model), `targets.py` / `paths.py` (target motion), `coverage.py` (map-coverage field), `tracker.py` / `repositioner.py` (the `make_tracker` / `make_repositioner` toggles + adapters), `testing.py` (batch/eval harness), `app.py` (dearpygui GUI), `visualize.py` (offline GIF/PNG). |
| **`evaluation/`** | **Ground-truth scorecard** (offline NumPy). `pcrlb.py` (bound), `gospa.py`, `ospa2.py`, `mot.py` (MOTA/IDF1), `hota.py`, `evaluate.py` (entry point). |
| **`diffsim/`** | **Differentiable core** (separate from GM-PHD). `model.py` (soft sensor), `rollout.py` (`jax.grad`-able uncertainty loss), `demo.py` (`uv run python -m diffsim.demo`), `config.py`. |

---

## Inherited legacy base (upstream packet-routing project — NOT used by the tracking sim)
The tracking sim imports only `NetCommWorld` (drone placement + linear kinematics) from here;
everything else is dead weight from the original UAV-mesh datagram-control project.

| Dir / file | What it was (legacy) |
|---|---|
| `netcomm/runner.py`, `types.py`, `visualizer.py` | Old per-packet episode loop, configs, plotting. Only `NetCommWorld` + `NetCommConfig` are reused. |
| `netcomm/routing/` (12) | Routing baselines (GPSR, AODV, DSR, OLSR, GNN/learning routers). |
| `netcomm/channel/` (8) | A2G channel: Doppler, path-loss, Nakagami, 3GPP LoS, SINR. |
| `netcomm/regime/` (5) | 4-state link hidden Markov model + filter. |
| `netcomm/world/` (6) | PPP node placement, kinematics, topology. |
| `netcomm/controller/`, `diversify/`, `packets/`, `lcb/`, `aoi/`, `beacons/`, `metrics/` | Per-packet decision policy, multipath diversification, packet queues, link-survival, age-of-information, beacons, comm metrics. |

---

## Supporting folders
| Dir | Contents |
|---|---|
| `experiments/` | CLI entry points. **Tracking:** `run_batch_tests.py`, `run_evaluation.py`, `run_tracking_demo.py`. The rest (`run_vop_validation.py`, `run_regime_*`, `run_baselines_ci.py`, `make_figures*.py`, …) are **legacy routing** experiments. |
| `baseline_papers/` | Source PDFs for the repositioning methods (Voronoi, MVMOT/infomax, non-myopic minimax). |
| `configs/` | YAML configs (`base.yaml`) — mostly legacy routing params. |
| `results/` | Output artifacts: `Repositioning/*.csv` (batch tables), `diffsim/diffsim_demo.png`, tracking images. |
| `tests/` | `test_jax_purity.py` — the JAX-purity CI gate (methods + sim-core + diffsim must use jit/vmap/scan). Per-package unit tests live in each package's own `tests/` dir. |

---

## Tests
Every active package ships a `tests/` dir (gmphd, modtrack, coverage_control, infomax, nonmyopic,
evaluation, diffsim, netcomm/tracking) plus the top-level purity gate. Run all:

```bash
JAX_PLATFORMS=cpu uv run --with pytest python -m pytest -q     # 234 passing
```

## The two-sentence summary
This is a **JAX multi-drone multi-object tracking simulator**: a GM-PHD tracker plus four
pluggable repositioners (Voronoi, greedy-MI, RSP, minimax), scored against ground truth, with a
GUI and a batch harness — that's the base. A separate `diffsim/` core provides a differentiable
(`jax.grad`-able) surrogate for gradient-based optimization; the folder also carries the
inherited upstream packet-routing project, which the tracking sim does not use.
