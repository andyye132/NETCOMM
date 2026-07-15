# NETCOMM Drone-Tracking Sim — Project State

_Last updated: 2026-06-22_

## 1. Purpose

A **JAX-based simulator for multi-drone, multi-object tracking**, built on the Tejadi/NETCOMM
packet-routing base. The sim knows the **true target states**, so it can benchmark existing
tracking and repositioning algorithms (drawn from papers) against ground truth.

Two pluggable roles, toggled independently:

- **Tracker** — estimates target states from drone observations. Default: **GM-PHD** (Vo & Ma 2006).
  Replaceable (e.g. by `modtrack`).
- **Repositioner** — moves the drones to improve tracking. Multiple implemented (below).
  Replaceable via a string toggle.

The two come together through **φ (the target/importance density)**: the tracker produces target
estimates → those estimates define φ → the repositioner optimizes drone placement against φ.

### Standing constraints (do not violate)
1. Implement methods in **JAX**.
2. Reference **CleanRL** (https://docs.cleanrl.dev/) and **PettingZoo** for code cleanliness.
3. Use **`uv`** for any package installs.
4. Methods must be **correct and exactly as described in their source paper**.
5. **Test correctness** against the paper's own results/figures (paper-faithful validation:
   replicate each paper's density/params/region for the correctness tests; φ is agnostic — paper
   density for tests, GM-PHD estimates for the sim).
6. **No git operations** unless the user explicitly asks.

## Session 2026-06-22 changes

Bug fixes and improvements landed this session (from a deep audit):

- **minimax** — fixed the Riccati recursion to match the paper's Eq. 7
  (predicted-to-predicted ordering).
- **GM-PHD** — fixed the component-merge covariance computation; converted the
  filter's hot kernels to JAX; added an optional **intensity-based birth mode**.
- **JAX conversion** — minimax, sensors, and coverage numeric kernels moved to
  JAX (jnp + jit/vmap on the hot paths).
- **`weight_by_phd`** — implemented (φ weighted by GM-PHD intensity).
- **Dead config removed** — unused infomax/minimax config fields pruned.
- **GUI** — Q4c relabel; Q5 Voronoi-cell overlay; Q6 metric toggles.
- **CI** — added a JAX-purity check.
- **`evaluation/`** — intentionally stays NumPy (offline, ground-truth scoring;
  not on the sim hot path).
- **`modtrack`** — intentionally left untouched: not wired in, gated by
  `NotImplementedError`, safe as long as it isn't selected as the tracker.

## 2. Repository layout

| Package / path | Role |
|---|---|
| `netcomm/tracking/` | Sim core: runner, GUI app, sensors, coverage field, repositioner adapters, test harness |
| `gmphd/` | GM-PHD filter (tracker) |
| `modtrack/` | Alternative tracker (drop-in for gmphd) |
| `coverage_control/` | Isotropic Voronoi coverage (Cortés 2004) — standalone package |
| `infomax/` | Greedy / RSP mutual-information planner (Corah & Michael, arXiv:2107.08550) |
| `nonmyopic/` | Non-myopic minimax target tracking (Zhang & Tokekar, arXiv:1611.02343) |
| `evaluation/` | Ground-truth scorecard: PCRLB, GOSPA, OSPA², MOT (MOTA/MOTP/IDF1), HOTA |
| `experiments/` | Headless CLI entry points (batch tests, evaluation, demo) |
| `results/Repositioning/` | CSV outputs from batch runs |

**Design pattern throughout:** each method is a *standalone package* (own config + algorithm +
paper-faithful tests) plus a *thin adapter* in `netcomm/tracking/repositioner.py` that implements
the `Repositioner` protocol. New methods plug in via `make_repositioner(name, cfg, sensor_cfg)`.

## 3. Methods implemented

### Trackers
- **GM-PHD** (`gmphd/`) — Gaussian-mixture PHD filter over a random finite set of targets.

### Repositioners (toggle strings in `_METHODS`)
| Toggle | Method | Source | Notes |
|---|---|---|---|
| `none` | Static drones | — | baseline |
| `isotropic_voronoi` | Voronoi/Lloyd coverage | Cortés 2004 | centroidal Voronoi descent; altitude preserved |
| `greedy_mi` | Sequential-greedy mutual information | Corah & Michael 2021 | ½-bound submodular; 2-step horizon |
| `rsp` | Randomized Sequential Partitions | Corah & Michael 2021 | RSP(n_d) + greedy |
| `minimax` | Non-myopic minimax `min_u max_z tr(Σ_T)` | Zhang & Tokekar 2016 | greedy-assignment multi-robot; α-β + redundancy pruning |

**Key correctness facts**
- Voronoi: φ = GM-PHD target estimates (sim) or paper density (tests); fixed-altitude ground plane;
  grid-quadrature centroids; Lyapunov descent proven in unit tests.
- Infomax: MI = `½ logdet(I + P·HᵀR⁻¹H)`; planning uses **ungated** `measurement_information`
  (`gate=False`) so there's a usable gradient outside the hard FOV cutoff. Gaussian MI in the sim,
  grid filter in the tests.
- Minimax: paper-exact worst-case-trace objective. Joint multi-robot minimax is intractable
  `((nk)^{RT})`, so we use the **authors' greedy-assignment route**. The ½ submodular bound does
  **not** transfer here (trace is only approximately supermodular). Pruning verified equal to full
  enumeration to 1e-9 with ~81% node reduction; redundancy pruning restricted to depth==1 for exactness.

## 4. Evaluation (`evaluation/`)

Ground-truth scorecard separating three qualities:
- **Repositioner quality** — `tracked_fraction`, PCRLB `bound_rmse`, cumulative `info_gain`.
- **Tracker quality** — `efficiency` (achieved RMS / bound RMS, both 2D-radial, frame-aligned).
- **Accuracy** — `GOSPA` (α=2), `OSPA²`, and CV-style `MOTA/MOTP/IDF1` + `HOTA`.

| Metric | Direction | Notes |
|---|---|---|
| GOSPA | lower better | α=2 only (canonical form); errors raise for other α |
| OSPA² | lower better | track-level, windowed |
| track% | higher better | fraction of targets tracked |
| efficiency | →1 (lower better) | RMS/RMS, frame-aligned |
| infogain | higher better | cumulative PCRLB info |
| achiev_rmse / bound_rmse | lower better | achieved vs PCRLB bound |
| MOTA / IDF1 / HOTA | higher better | CV metrics; computed from associations, no actual vision |
| MOTP | lower better | mean matched distance |

PCRLB uses the information-filter recursion `J_k = (F J_{k-1}⁻¹ Fᵀ + Q)⁻¹ + HᵀMH`; incorporates
target velocity/acceleration via the CV or CA motion model (`eval_motion`). LoS/NLoS derived from
horizontal distance + elevation angle.

## 5. Test / evaluation harness

- **`netcomm/tracking/testing.py`** — `run_test` (N epochs, parallel via multiprocessing spawn,
  per-metric mean/std/min/max/median) and `run_batch` (sweep methods × rounds × epochs with
  **matched paired seeds**: within a round every method runs the *identical* scenarios — same random
  drone starts, same target motion — for a fair paired comparison). Writes CSV to
  `results/Repositioning/<name>.csv`. `TOP_METRICS` = GOSPA, track%, OSPA², effic, infogain,
  achiev_rmse, bound_rmse.
- **CLI:** `python -m experiments.run_batch_tests --methods ... --epochs --rounds --batch-size ...`
- **GUI:** "Single Method Test" and "Batch Eval" tabs call the same harness functions.

## 6. GUI (`netcomm/tracking/app.py`, dearpygui)

Top **tab bar** with four modes:
- **Custom** — manual scenario building (collapsing headers, default closed: Drones / Objects /
  Heat map / Tracking / Repositioning / Simulation). Place drones/objects on the canvas; Run / Play /
  Reset / scrub / toggle true-vs-estimated view.
- **Single Method Test** — pick method/tracker/motion/epochs/drones/objects → score one method.
- **Multi-Screen Test** — split-canvas, same scenario / different methods. **Stub (planned).**
- **Batch Eval** — toggle batch size, methods (checkboxes), epochs/rounds → headless table + CSV.

**Recent GUI work (this session):**
- Run executes in a **background thread**; button disables + shows "Running…"; a **progress bar**
  fills per step (no more spam-clicking / UI freeze).
- **JIT warmup** at startup (background thread) removes the first-click compile delay.
- **Live graphs** (left panel): map-coverage uncertainty, **target track σ** (mean GM-PHD track
  uncertainty — the one that actually drops as targets get tracked), and track count.
- Visual random-drone preview; random-objects slider hidden from canvas; dropdowns start closed;
  tracking/repositioning toggleable.

### Note on the "uncertainty stuck >95%" question
The "MAP COVERAGE UNCERTAINTY" headline is `1 − mean_certainty` over the **whole-map** CoverageField
(`coverage.py`): a 64×64 grid that decays everywhere and is restored only under drone footprints.
A footprint radius ≈ `altitude·tan(fov)` covers ~1.4% of a 300×300 map per drone, so the grid mean
stays low and uncertainty stays ~95–98%. **This is a whole-map coverage metric, not target-tracking
quality** — for tracking quality, read the new track-σ plot.

## 7. Status

- **Tests:** 189 passing (`JAX_PLATFORMS=cpu python -m pytest -q`).
- **Working end-to-end:** all four repositioners run in-sim and through the harness; evaluation
  scorecard; batch CSV; threaded GUI with live graphs + progress bar + JIT warmup.

### Known limitations / open items
- **Multi-Screen (split-canvas)** — stubbed. Plan: parameterize `draw_scene` to a drawlist tag +
  sub-rectangle; run 2–4 same-scenario sims with different methods; synced playback with per-pane
  score captions. Moderate effort.
- **Speed** — biggest remaining win is **vectorizing `measurement_information`** (currently called in
  Python loops per drone×target inside greedy_mi/minimax planning). Other knobs: smaller action
  sets / horizon, lower coverage-grid resolution for the live view.
- **Quick wins not yet done:** export GIF/PNG/CSV from the GUI, save/load scenario presets, tooltips/legend.
- MCTS noted as a reusable anytime planner but not fully factored into its own module.

## 8. How to run

```bash
# GUI
python -m netcomm.tracking.app

# Headless batch (table + CSV to results/Repositioning/)
python -m experiments.run_batch_tests --methods none isotropic_voronoi greedy_mi rsp \
    --epochs 20 --rounds 3 --batch-size 8 --drones 5 --targets 6 --motion circle --name circle_sweep

# Tests
JAX_PLATFORMS=cpu python -m pytest -q
```

## 9. Memory / context files
- `memory/MEMORY.md` — index of persistent notes.
- `memory/netcomm-tracking-sim-goal.md` — what the sim is for; how methods plug in.
- `memory/paper-faithful-validation.md` — replicate each paper's own setup for correctness tests.
- `memory/evaluation-metric.md` — ground-truth scorecard design (PCRLB + GOSPA + OSPA²).
