# NETCOMM: Regime-Adaptive Datagram Control

Per-packet belief-state stochastic controller for deadline-aware robot networks under hidden Doppler fading. For each packet the controller picks one of `{react, predict, diversify, drop}` by maximizing deadline utility under a 4-state per-link HMM belief over `{stable, predictable, volatile, blocked}`.

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
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

JAX (CPU or CUDA) is required.

## Smoke

```bash
python -m experiments.smoke
```

## Full sweep

```bash
python -m experiments.run_baselines_ci
python -m experiments.run_regime_sweep
python -m experiments.run_ablations
python -m experiments.run_vop_validation
python -m experiments.run_vod_validation
python -m experiments.run_mode_occupancy
python -m experiments.run_calibration
python -m experiments.run_scalability
python -m experiments.run_hmm_inference
python -m experiments.run_robustness
python -m experiments.run_overhead
python -m experiments.run_udp_stress

python -m experiments.make_figures
python -m experiments.make_tables
```

# NETCOMM
