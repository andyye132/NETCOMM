# NETCOMM: Regime-Adaptive Datagram Control

Per-packet belief-state stochastic controller for deadline-aware robot networks under hidden Doppler fading. For each packet the controller picks one of `{react, predict, diversify, drop}` by maximizing deadline utility under a 4-state per-link HMM belief over `{stable, predictable, volatile, blocked}`.

The paper is at [netcomm.tex](netcomm.tex). The implementation spec is at [netcomm_impl.md](netcomm_impl.md).

## Layout

```
netcomm/              new package
  types.py            shared NamedTuples + ControllerProtocol (sync point)
  world/              PPP, kinematics, environment, topology, node state
  channel/            doppler, path-loss, Nakagami-m, 3GPP LoS, SINR, forecast, linkup
  regime/             per-link 4-state HMM (filter, observations, transitions, oracle)
  lcb/                lower-confidence-bound link survival
  controller/         utility estimators (react / predict / diversify / drop) + decide
  diversify/          k-disjoint paths, k-of-n decode, greedy + Sinkhorn fragment allocator
  packets/            Packet dataclass, priority queue, Poisson/bursty generator
  routing/            baseline policies (GPSR, GLSR, AODV, DSR, OLSR, P-OLSR, TGPSR, P3,
                      CAR, learning, GNN), oracle, PerPacketHMMController, always-* ablations
  aoi/                Age-of-Information tracker
  beacons/            adaptive beacon cadence
  metrics/            delivery, AoI, runtime, route churn, Brier/ECE calibration, mode occupancy
  runner.py           per-packet episode loop
  visualizer.py       4-state belief overlay + per-packet mode coloring

experiments/          12 sweep drivers + smoke + sbatch wrappers + make_figures + make_tables
configs/              base.yaml + 5 scenarios + ~20 method YAMLs
results/              parquets + figures + animations (generated, ignored by git)
ns3_validation/       NS-3 cross-validation scaffolding for the new controller

legacy/               archived OLD stack (BFS predictor + Wonham SAGIN tier; pre-rewrite).
                      Still runnable from inside legacy/; not on the critical path.
                      See legacy/README.md.
```

## Install

```bash
source /users/aiyer40/TRIAGE/.venv/bin/activate    # shared venv on this machine
pip install -e .                                    # installs netcomm
```

JAX (CPU or CUDA) is required. On OSCAR the venv at the path above already has JAX 0.6.2.

## Smoke

Login-node Python is forbidden. Use an interactive GPU node:

```bash
srun --pty -p gpu --gres=gpu:1 --mem=16G --time=00:15:00 \
  bash -c "source /users/aiyer40/TRIAGE/.venv/bin/activate && \
           cd /users/aiyer40/NETCOMM && python -m experiments.smoke"
```

The smoke runs 1 scenario × 3 seeds × {adaptive, GPSR, always-predict} for 50 steps and asserts: no NaN/Inf in `regime_belief`; all four action values appear; mean delivery in `(0, 1)`; per-packet runtime under 50 ms at N=16.

## Full sweep

```bash
for s in baselines_ci regime_sweep ablations vop_validation vod_validation \
         mode_occupancy calibration scalability hmm_inference robustness \
         overhead udp_stress; do
  sbatch experiments/sbatch_${s}.sh
done

# After all parquets exist:
srun --pty -p gpu --gres=gpu:1 --mem=16G --time=00:30:00 \
  bash -c "source /users/aiyer40/TRIAGE/.venv/bin/activate && cd /users/aiyer40/NETCOMM && \
           python -m experiments.make_figures && python -m experiments.make_tables"

pdflatex netcomm.tex
```

OSCAR QOS caps concurrent GPU jobs at 2. Total full-scope sweep ~360 GPU-hours, ~4–6 weeks wall-clock.

## Tests + figures map

| Test (impl spec § 12) | Script | Parquet | Figure |
|---|---|---|---|
| 1 Regime sweep | `run_regime_sweep` | `results/regime_sweep/` | Fig 5 |
| 2 VoP validation | `run_vop_validation` | `results/vop_validation/` | Fig 7 |
| 3 VoD validation | `run_vod_validation` | `results/vod_validation/` | Fig 8 |
| 4 Calibration | `run_calibration` | `results/calibration/` | Fig 9 |
| 5 HMM inference | `run_hmm_inference` | `results/hmm_inference/` | (text) |
| 6 Robustness | `run_robustness` | `results/robustness/` | (text) |
| 7 Overhead | `run_overhead` | `results/overhead/` | (text) |
| 8 Scalability | `run_scalability` | `results/scalability/` | Fig 12 / Tab 4 |
| 9 UDP stress | `run_udp_stress` | `results/udp_stress/` | (text) |
| 10 Oracle / baselines | `run_baselines_ci` | `results/baselines/` | Fig 10 / Tab 3 |
| Ablations | `run_ablations` | `results/ablations/` | Fig 11 / Tab 5 |
| Mode occupancy | `run_mode_occupancy` | `results/mode_occupancy/` | Fig 6 |

Figs 1, 2 are TikZ / schematic (in `netcomm.tex`); Figs 3, 4 are produced from a smoke snapshot.

## Legacy

The OLD predictor + Wonham SAGIN stack is archived under [legacy/](legacy/). Do not import from it at runtime; the new package is fully self-contained.
# NETCOMM
