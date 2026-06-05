# NETCOMM Results Index

This file maps every paper figure and table to the script that produces it
and the parquet it reads.

All experiments run via `sbatch experiments/sbatch_<name>.sh`. Figures and
tables are pure-Python and may run on the login node.

## Pipeline

1. Run experiments: `sbatch experiments/sbatch_<name>.sh` (one per row below).
2. Build figures: `python -m experiments.make_figures` -> `results/figures/*.pdf`.
3. Build tables:  `python -m experiments.make_tables`  -> `results/tables/*.tex`.

## Figures

| Fig | Script                              | Parquet                              |
|----:|-------------------------------------|--------------------------------------|
| 1   | TikZ (paper)                        | -- (schematic)                       |
| 2   | TikZ (paper)                        | -- (block diagram)                   |
| 3   | TODO Agent 1 snapshot               | -- (single representative trace)     |
| 4   | TODO controller-grid sweep          | -- (decision boundary)               |
| 5   | `run_regime_sweep.py`               | `results/regime_sweep/sweep.parquet` |
| 6   | `run_mode_occupancy.py`             | `results/mode_occupancy/sweep.parquet` |
| 7   | `run_vop_validation.py`             | `results/vop_validation/sweep.parquet` |
| 8   | `run_vod_validation.py`             | `results/vod_validation/sweep.parquet` |
| 9   | `run_calibration.py`                | `results/calibration/sweep.parquet`  |
| 10  | `run_baselines_ci.py`               | `results/baselines/sweep.parquet`    |
| 11  | `run_ablations.py`                  | `results/ablations/sweep.parquet`    |
| 12  | `run_scalability.py`                | `results/scalability/sweep.parquet`  |

## Tables

| Tab | Source                              | Parquet                              |
|----:|-------------------------------------|--------------------------------------|
| 1   | static registry in `make_tables.py` | --                                   |
| 2   | reads `configs/scenarios/*.yaml`    | --                                   |
| 3   | `run_baselines_ci.py`               | `results/baselines/sweep.parquet`    |
| 4   | `run_scalability.py`                | `results/scalability/sweep.parquet`  |
| 5   | `run_ablations.py`                  | `results/ablations/sweep.parquet`    |

## Tests (impl plan section 12)

| Test | Script                              |
|-----:|-------------------------------------|
| 1    | `run_regime_sweep.py`               |
| 2    | `run_vop_validation.py`             |
| 3    | `run_vod_validation.py`             |
| 4    | `run_calibration.py`                |
| 5    | `run_hmm_inference.py`              |
| 6    | `run_robustness.py`                 |
| 7    | `run_overhead.py`                   |
| 8    | `run_scalability.py`                |
| 9    | `run_udp_stress.py`                 |
| 10   | `run_baselines_ci.py` (with `oracle_future`) |

## Smoke check

```
srun --pty -p gpu --gres=gpu:1 --time=00:30:00 \
  bash -c "source /users/aiyer40/TRIAGE/.venv/bin/activate && \
           cd /users/aiyer40/NETCOMM && python -u -m experiments.smoke"
```
