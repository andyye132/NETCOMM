# NETCOMM Drone-Tracking Sim — Base & Demo Notes

_The working foundation: a JAX multi-drone, multi-object tracking simulator you can run
different **tracking** and **repositioning** methods on, scored against ground truth._

## What it is (one paragraph)
N drones (downward cameras) observe M moving ground targets. A **tracker** estimates target
states from noisy detections; a **repositioner** moves the drones to track better. The two
roles toggle independently and couple through **φ** (the importance density: the tracker's
belief tells the repositioner where the targets are). Because the sim knows the **true** target
states, every method is benchmarked against ground truth.

**Extension points (this is the "base"):** add a tracker in `make_tracker(name)`
(`netcomm/tracking/tracker.py`); add a repositioner in `make_repositioner(name)`
(`netcomm/tracking/repositioner.py`). Each method is a standalone package + a thin adapter.

## How to run
| | command |
|---|---|
| GUI | `uv run python -m netcomm.tracking.app` |
| Headless batch (table + CSV) | `uv run python -m experiments.run_batch_tests --methods none isotropic_voronoi greedy_mi rsp minimax --epochs 20 --rounds 3` |
| Differentiable-core demo | `uv run python -m diffsim.demo` |
| Tests (234 pass) | `JAX_PLATFORMS=cpu uv run --with pytest python -m pytest -q` |

## Trackers — "where are the targets?"
- **GM-PHD** (default, `gmphd/`, Vo & Ma 2006). A Gaussian-Mixture Probability Hypothesis
  Density filter: tracks an unknown, time-varying number of targets as a random finite set,
  with birth/death and clutter, **no hard data association**. All drones' detections are pooled
  into one centralized filter. Output per target: mean + covariance (the covariance is the
  tracking-uncertainty signal the GUI plots). *Explain: "a principled multi-target Bayesian
  filter — it figures out how many targets there are and where, from noisy detections."*
- _modtrack — fusion primitives only, not wired in; ignore._

## Repositioners — "where should the drones go?" (toggle string)
- **none** — drones stay put (baseline).
- **isotropic_voronoi** (`coverage_control/`, Cortés 2004). Each drone owns a φ-weighted
  Voronoi cell and moves toward its cell's centroid (Lloyd / centroidal-Voronoi descent).
  *Explain: "coverage control — drones spread to cover where the targets are. The gold dot is
  the cell centroid each drone steers toward."*
- **greedy_mi** (`infomax/`, Corah & Michael 2021). Sequential-greedy **mutual information**:
  drones choose actions one at a time, each maximizing its marginal info gain about the targets
  given the others' choices. *Explain: "information-greedy — drones go where they most reduce
  target uncertainty, coordinating so they don't redundantly cover the same target."*
- **rsp** (same package). Randomized Sequential Partitions — greedy_mi but drones plan in
  parallel rounds. *Explain: "a scalable/parallel version of greedy-MI."*
- **minimax** (`nonmyopic/`, Zhang & Tokekar 2016). Non-myopic **worst-case**: each drone plans
  to minimize the worst-case future tracking covariance (min over its moves, max over
  measurement outcomes); drones are assigned to targets by sequential greedy. *Explain:
  "robust / non-myopic — plans against the worst case, for when you can't afford to lose a
  target."*

## Sensor model (`netcomm/tracking/sensors.py`)
Downward camera at altitude h: circular ground footprint of radius **h·tan(FOV)**. A target
inside is detected with prob `p_detect`, returning a noisy position + covariance **R** that
grows with slant range / off-nadir angle (fusing two drones from different bearings shrinks the
error ellipse). Targets **outside** the footprint are not seen. Optional LoS/NLoS elevation
model (Alzenad & Yanikomeroglu 2018, dense-urban) is off by default.

## Evaluation (`evaluation/`) — scored vs ground truth
- **GOSPA** ↓ — multi-target localization + cardinality error.
- **OSPA²** ↓ — track-level error over time windows.
- **PCRLB bound_rmse** ↓ — information-theoretic best-possible error for the realized drone
  geometry (**repositioner** quality).
- **efficiency** = achieved/bound (→1) — **tracker** quality.
- **tracked fraction** ↑, **info gain** ↑, plus CV metrics **MOTA / IDF1 / HOTA**.
- You can toggle which metrics to show before a Single/Batch run.

## GUI cheat-sheet
- **Tabs:** Custom (place + run a scenario), Single (score one method over N epochs), Batch
  (compare methods → table + CSV). Multi-screen tab is a stub.
- **Yellow dot + line** = the Voronoi cell centroid each drone steers toward (isotropic_voronoi
  only). **Show Voronoi cells** = the colored cell partition (auto-on when you pick voronoi).
- **Two uncertainty readouts:** **TARGET TRACK UNCERTAINTY** = mean GM-PHD track σ — the real
  tracking-quality signal (drops as targets are tracked). **AREA COVERAGE** = fraction of the
  *whole map* unobserved — a coverage metric that stays high; **not** a tracking failure.
- Drones only *see* objects in their footprint (green ring = currently seen). The map draws all
  true objects for **you**, the viewer — that's not what the drones know.

## Differentiable core (`diffsim/`) — separate from GM-PHD
A smooth, fully-JAX surrogate of the same physics (soft FOV gate + per-target Gaussian/Kalman
belief) whose tracking-uncertainty loss is **`jax.grad`-able**. Backprop through a whole episode
to optimize drone placement by gradient descent. `uv run python -m diffsim.demo` → finite-diff
check (grad correct to ~1e-7) + ~99% uncertainty reduction + a figure in `results/diffsim/`.
*Explain: "the main sim uses GM-PHD, which isn't differentiable; this is a differentiable core
for gradient-based optimization / learning of repositioning. We verify it with finite-difference
gradient checking."*

## State & honest caveats
- **234 tests pass.** A JAX-purity gate enforces the methods + sim-core + diffsim are genuine JAX.
- **Differentiability = the `diffsim` core only.** The GM-PHD sim is JAX-*accelerated*, not
  differentiable (variable target count + argmax + hard FOV gate + stochastic detection).
- greedy_mi/rsp use exhaustive single-robot planning (paper's MCTS not built; exhaustive is exact
  for the small action set). minimax's multi-robot layer is the authors' greedy extension. Both
  are wired and tested.
- modtrack is a stub (not selectable). `evaluation/` is intentionally NumPy (offline analysis).
- **Recently fixed:** target-spawn bug (objects were glued to drones — also inflated benchmark
  scores), minimax Riccati Eq-7 ordering, GM-PHD merge covariance, non-PSD guard, Voronoi cell
  view, OSPA²/LoS paper-fidelity tests, and the missing `dearpygui` dependency.
