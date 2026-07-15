"""Evaluation-test harness: score a (tracker, repositioner) method over Monte-Carlo epochs.

An "epoch" is one randomized sim episode (random drone placement + a moving-target scenario)
run for n_steps and scored by evaluation.evaluate(). run_test() runs many epochs (optionally
in parallel across CPU cores) and aggregates each metric's mean/std/min/max/median. run_batch()
sweeps several methods, prints a table, and writes a CSV to results/Repositioning/.

Shared by the headless CLI (experiments/run_batch_tests.py) and the GUI "Run Test" button.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")              # CPU in this process + spawned workers

import multiprocessing as mp
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# (evaluate summary key, display name, lower_is_better)
TOP_METRICS = [
    ("gospa_mean", "GOSPA", True),
    ("tracked_fraction", "track%", False),
    ("ospa2_mean", "OSPA2", True),
    ("efficiency_mean", "effic", True),
    ("info_gain_cumulative", "infogain", False),
    ("achieved_rmse_mean", "achiev_rmse", True),
    ("bound_rmse_mean", "bound_rmse", True),
]
STATS = ("mean", "std", "min", "max", "median")

# Full set of selectable metrics (superset of TOP_METRICS) drawn from the evaluate()
# summary. The GUI metric-toggle UI (Q6) picks from these; run_test/run_batch/format_table
# accept an optional ``metrics`` list of summary keys (default None = TOP_METRICS).
# (summary key, display name, lower_is_better). CV/MOT keys (mota/idf1/hota/...) require
# compute_mot=True; _needs_mot() detects them so the harness flips it on automatically.
METRIC_REGISTRY = [
    ("gospa_mean", "GOSPA", True),
    ("gospa_localization_mean", "gospa_loc", True),
    ("tracked_fraction", "track%", False),
    ("ospa2_mean", "OSPA2", True),
    ("efficiency_mean", "effic", True),
    ("info_gain_cumulative", "infogain", False),
    ("info_gain_per_step", "infogain/step", False),
    ("achieved_rmse_mean", "achiev_rmse", True),
    ("bound_rmse_mean", "bound_rmse", True),
    ("bound_rmse_observed", "bound_rmse_obs", True),
    ("missed_total", "missed", True),
    ("false_total", "false", True),
    ("n_est_tracks", "n_tracks", False),
    # CV / vision MOT metrics (need compute_mot=True)
    ("mota", "MOTA", False),
    ("motp", "MOTP", True),
    ("idf1", "IDF1", False),
    ("id_switches", "id_sw", True),
    ("hota", "HOTA", False),
    ("deta", "DetA", False),
    ("assa", "AssA", False),
]
_REGISTRY_BY_KEY = {key: (key, disp, lower) for key, disp, lower in METRIC_REGISTRY}
# CV / vision MOT summary keys: selecting any of these forces EvalConfig(compute_mot=True).
_MOT_KEYS = frozenset({"mota", "motp", "idf1", "id_switches", "hota", "deta", "assa"})


def available_metrics():
    """The full selectable metric list (key, display, lower_is_better) for the GUI Q6 UI."""
    return list(METRIC_REGISTRY)


def _resolve_metrics(metrics: Optional[Sequence[str]]):
    """Normalize a caller's metric selection into the (key, display, lower) tuples used
    throughout the harness. None -> the default TOP_METRICS. Unknown keys are dropped;
    an empty/all-unknown selection falls back to TOP_METRICS so callers never see nothing."""
    if metrics is None:
        return list(TOP_METRICS)
    resolved = [_REGISTRY_BY_KEY[k] for k in metrics if k in _REGISTRY_BY_KEY]
    return resolved or list(TOP_METRICS)


def _needs_mot(metric_tuples) -> bool:
    """True if any selected metric is a CV/vision MOT scalar (=> compute_mot must be on)."""
    return any(key in _MOT_KEYS for key, _, _ in metric_tuples)


@dataclass
class TestConfig:
    __test__ = False                        # not a pytest test class despite the name
    method: str = "isotropic_voronoi"       # repositioner toggle
    tracker: str = "gmphd"
    n_drones: int = 4                       # randomly placed each epoch
    drone_radius: float = 22.0              # footprint radius for placement
    target_motion: str = "random_walk"      # random_walk | figure8 | circle | square | triangle
    n_targets: int = 5
    n_steps: int = 60
    dt: float = 0.2
    area: tuple = (0.0, 300.0, 0.0, 300.0)
    object_speed: float = 5.0
    fov_deg: float = 55.0
    los_nlos: bool = False
    eval_motion: str = "cv"                 # PCRLB target model (cv | ca)
    repos_cfg: object = None                # method config (auto-filled by default_repos_cfg)


def default_repos_cfg(method: str):
    """A sensible default config object for a repositioner method (None for none/voronoi)."""
    key = (method or "none").lower()
    if key in ("greedy_mi", "rsp"):
        from infomax import InfomaxConfig
        return InfomaxConfig(method="greedy" if key == "greedy_mi" else "rsp")
    if key == "minimax":
        from nonmyopic import MinimaxConfig
        return MinimaxConfig()              # explicit (not an accidental None fallthrough)
    return None                             # none / isotropic_voronoi use their own defaults


def _sensor(cfg: TestConfig):
    from netcomm.tracking import CameraSensorConfig
    return CameraSensorConfig(half_fov_rad=np.deg2rad(cfg.fov_deg), los_nlos=cfg.los_nlos)


def run_episode(cfg: TestConfig, seed: int,
                metrics: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """One epoch: randomly place drones, run the scenario, evaluate. Returns the selected
    metrics (default None = TOP_METRICS). compute_mot is turned on automatically if any
    selected metric is a CV/vision MOT scalar (mota/idf1/hota/...)."""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # benign empty-slice / nan stats
    from netcomm.tracking import run_preset_tracking
    from evaluation import evaluate, EvalConfig

    metric_tuples = _resolve_metrics(metrics)
    rng = np.random.default_rng(seed)
    xmn, xmx, ymn, ymx = cfg.area
    drones = [(float(rng.uniform(xmn, xmx)), float(rng.uniform(ymn, ymx)), cfg.drone_radius)
              for _ in range(cfg.n_drones)]
    sensor = _sensor(cfg)
    res = run_preset_tracking(
        drones, cfg.target_motion, n_objects=cfg.n_targets, n_steps=cfg.n_steps, dt=cfg.dt,
        area_xy=cfg.area, sensor_cfg=sensor, object_speed=cfg.object_speed, tracker=cfg.tracker,
        repositioner=cfg.method, repos_cfg=cfg.repos_cfg, seed=seed)
    summary = evaluate(res, sensor,
                       EvalConfig(motion=cfg.eval_motion, compute_mot=_needs_mot(metric_tuples)),
                       dt=cfg.dt)["summary"]
    return {key: float(summary.get(key) if summary.get(key) is not None else np.nan)
            for key, _, _ in metric_tuples}


def _worker(args):
    cfg, seed, metrics = args
    try:
        return run_episode(cfg, seed, metrics)
    except Exception as e:                  # keep the batch alive; drop failed epochs
        return {"_error": f"{type(e).__name__}: {e}"}


def _run_episodes(cfg: TestConfig, seeds: Sequence[int], n_workers: int,
                  progress: Optional[Callable[[int, int], None]] = None,
                  metrics: Optional[Sequence[str]] = None):
    """Run the episodes at the given seeds (parallel if n_workers>1). Returns (results, errors)."""
    if cfg.repos_cfg is None:
        cfg = replace(cfg, repos_cfg=default_repos_cfg(cfg.method))
    if n_workers and n_workers > 1 and len(seeds) > 1:
        with mp.get_context("spawn").Pool(min(int(n_workers), len(seeds))) as pool:
            results = pool.map(_worker, [(cfg, s, metrics) for s in seeds])
    else:
        results = []
        for i, s in enumerate(seeds):
            results.append(_worker((cfg, s, metrics)))
            if progress:
                progress(i + 1, len(seeds))
    errors = [r["_error"] for r in results if "_error" in r]
    return [r for r in results if "_error" not in r], errors


def _aggregate(results: List[Dict], metric_tuples=TOP_METRICS) -> Dict[str, Dict[str, float]]:
    """Per-metric mean/std/min/max/median over a list of episode metric dicts."""
    stats: Dict[str, Dict[str, float]] = {}
    for key, _, _ in metric_tuples:
        vals = np.array([r[key] for r in results if key in r], dtype=float) if results \
            else np.array([])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            stats[key] = {"mean": float(vals.mean()), "std": float(vals.std()),
                          "min": float(vals.min()), "max": float(vals.max()),
                          "median": float(np.median(vals))}
        else:
            stats[key] = {s: float("nan") for s in STATS}
    return stats


def run_test(cfg: TestConfig, n_epochs: int = 10, n_workers: int = 1, seed_base: int = 0,
             progress: Optional[Callable[[int, int], None]] = None,
             metrics: Optional[Sequence[str]] = None) -> Dict:
    """Run n_epochs episodes (seeds seed_base..+n) and aggregate per-metric stats.

    ``metrics`` (optional) is a list of evaluate() summary keys to score; None keeps the
    default TOP_METRICS. The aggregated stats dict is keyed by whatever was selected."""
    metric_tuples = _resolve_metrics(metrics)
    seeds = [seed_base + i for i in range(n_epochs)]
    results, errors = _run_episodes(cfg, seeds, n_workers, progress, metrics)
    if errors:
        import warnings
        warnings.warn(f"run_test({cfg.method}): {len(errors)}/{n_epochs} epochs FAILED and "
                      f"were dropped from the stats (first: {errors[0]})", RuntimeWarning,
                      stacklevel=2)
    return {"method": cfg.method, "n_epochs": len(results),
            "stats": _aggregate(results, metric_tuples), "raw": results, "errors": errors,
            "metrics": metric_tuples}


def run_batch(methods: Sequence[str], cfg: TestConfig, n_epochs: int = 10, n_rounds: int = 1,
              batch_size: int = 1, out_dir: Optional[Path] = None, name: str = "batch",
              progress: Optional[Callable] = None,
              metrics: Optional[Sequence[str]] = None,
              repos_cfgs: Optional[Dict[str, object]] = None):
    """Sweep methods over n_rounds x n_epochs with MATCHED seeds, aggregate, write a CSV.

    Each round r uses a fresh seed-set, but WITHIN a round every method runs the IDENTICAL
    scenarios (same random drone starts + same target movements) for a paired comparison.
    Reports each metric's mean/std over all method-epochs plus the round-to-round std of the
    per-round mean (the variance of the score estimate). CSV -> results/Repositioning/<name>.csv.

    ``metrics`` (optional) is a list of evaluate() summary keys; None keeps TOP_METRICS. The
    rows / CSV columns carry exactly the selected metrics (each as <disp>_<stat> + _round_std).

    ``repos_cfgs`` (optional) maps method name -> custom config object for that method,
    overriding the defaults (so a tuned config CAN be A/B-ed inside a sweep). A method not
    in the map falls back to cfg.repos_cfg (if cfg.method matches) and then to
    default_repos_cfg(method).
    """
    metric_tuples = _resolve_metrics(metrics)

    def _cfg_for(m: str):
        if repos_cfgs is not None and m in repos_cfgs:
            return repos_cfgs[m]
        if cfg.repos_cfg is not None and m == cfg.method:
            return cfg.repos_cfg
        return default_repos_cfg(m)

    acc: Dict[str, List[Dict]] = {m: [] for m in methods}        # all episodes across rounds
    round_means: Dict[str, List[Dict]] = {m: [] for m in methods}
    n_failed: Dict[str, int] = {m: 0 for m in methods}
    first_error: Dict[str, Optional[str]] = {m: None for m in methods}
    for r in range(n_rounds):
        seed_base = (r + 1) * 1_000_000                          # distinct per round, shared across methods
        seeds = [seed_base + i for i in range(n_epochs)]
        for m in methods:
            c = replace(cfg, method=m, repos_cfg=_cfg_for(m))
            res, err = _run_episodes(c, seeds, batch_size, metrics=metrics)
            acc[m].extend(res)
            n_failed[m] += len(err)
            if err and first_error[m] is None:
                first_error[m] = err[0]
            round_means[m].append({key: float(np.nanmean([d[key] for d in res])) if res else float("nan")
                                   for key, _, _ in metric_tuples})
        if progress:
            progress(r + 1, n_rounds)

    rows: List[Dict] = []
    for m in methods:
        if n_failed[m]:
            import warnings
            warnings.warn(f"run_batch({m}): {n_failed[m]} epoch(s) FAILED and were dropped "
                          f"(first: {first_error[m]})", RuntimeWarning, stacklevel=2)
        stats = _aggregate(acc[m], metric_tuples)
        row = {"method": m, "epochs": n_epochs, "rounds": n_rounds, "n": len(acc[m]),
               "n_failed": n_failed[m]}
        for key, disp, _ in metric_tuples:
            for s in STATS:
                row[f"{disp}_{s}"] = stats[key][s]
            rm = [d[key] for d in round_means[m] if np.isfinite(d[key])]
            row[f"{disp}_round_std"] = float(np.std(rm)) if len(rm) > 1 else 0.0
        rows.append(row)

    out_dir = Path(out_dir) if out_dir is not None else (REPO / "results" / "Repositioning")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    _write_csv(rows, csv_path)
    return rows, csv_path


def _write_csv(rows: List[Dict], path: Path) -> None:
    import csv
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def format_table(rows: List[Dict], metrics: Optional[Sequence[str]] = None) -> str:
    """Pretty mean+/-std table (one row per method, one column per selected metric).

    ``metrics`` (optional) is a list of evaluate() summary keys; None keeps TOP_METRICS. Only
    columns actually present on the rows are shown, so this stays compatible with rows built
    by an earlier run_batch metric selection."""
    metric_tuples = _resolve_metrics(metrics)
    if rows:                                # only show columns the rows actually carry
        metric_tuples = [t for t in metric_tuples if f"{t[1]}_mean" in rows[0]]
    hdr = (f"{'method':<18}{'epochs':>7}{'rounds':>7}  "
           + "".join(f"{disp:>16}" for _, disp, _ in metric_tuples))
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        cells = "".join(f"{r[f'{disp}_mean']:>9.2f}±{r[f'{disp}_std']:<6.2f}"
                        for _, disp, _ in metric_tuples)
        lines.append(f"{r['method']:<18}{r.get('epochs', r.get('n_epochs', 0)):>7}"
                     f"{r.get('rounds', 1):>7}  " + cells)
    lower = [disp for _, disp, low in metric_tuples if low]
    higher = [disp for _, disp, low in metric_tuples if not low]
    lines.append(f"\nlower is better: {', '.join(lower) or '-'};  "
                 f"higher is better: {', '.join(higher) or '-'}.")
    return "\n".join(lines)
