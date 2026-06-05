import os
from pathlib import Path
from datetime import date
from typing import Tuple, List, Optional, Dict, Any

import yaml
import pandas as pd

from netcomm.types import NetCommConfig, ControllerProtocol

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_scenario_method(scn_name: str, mth_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base = _load_yaml(CFG_DIR / "base.yaml")
    scn = _load_yaml(CFG_DIR / "scenarios" / f"{scn_name}.yaml")
    mth = _load_yaml(CFG_DIR / "methods" / f"{mth_name}.yaml")
    merged = dict(base)
    merged.update(scn)
    merged["_scenario_name"] = scn_name
    return merged, mth


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

_NETCOMM_FIELDS = NetCommConfig._fields  # introspect NamedTuple
_NETCOMM_DEFAULTS = NetCommConfig._field_defaults


def build_cfg(scn_dict: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> NetCommConfig:
    d = dict(scn_dict)
    if overrides:
        d.update(overrides)

    def _maybe_tuple(v):
        if isinstance(v, list):
            return tuple(v)
        return v

    kwargs: Dict[str, Any] = {}
    for f in _NETCOMM_FIELDS:
        if f in d:
            kwargs[f] = _maybe_tuple(d[f])
        elif f in _NETCOMM_DEFAULTS:
            kwargs[f] = _NETCOMM_DEFAULTS[f]
        else:
            raise KeyError(f"NetCommConfig field '{f}' missing from scenario dict")

    # coerce scalar numeric fields to their expected types
    def _f(name): kwargs[name] = float(kwargs[name])
    def _i(name): kwargs[name] = int(kwargs[name])
    for name in ("f_c", "bandwidth", "p_tx", "n0", "gamma_th", "r_rng", "T_AoI",
                 "pi_min", "alpha_pl", "beta_pl", "m_0", "T_0", "cluster_kappa",
                 "lambda_density", "dt", "lambda_B", "lambda_E", "lambda_C",
                 "lambda_Q", "z_delta"):
        _f(name)
    for name in ("n_horizon_steps", "n_nodes", "k_paths", "n_fragments",
                 "k_decode", "priority_classes", "compute_budget_per_step"):
        _i(name)
    kwargs["env"] = str(kwargs["env"])
    kwargs["area_xy"] = tuple(float(x) for x in kwargs["area_xy"])
    kwargs["z_range"] = tuple(float(x) for x in kwargs["z_range"])
    kwargs["pkt_rate_per_class"] = tuple(float(x) for x in kwargs["pkt_rate_per_class"])
    return NetCommConfig(**kwargs)


# ---------------------------------------------------------------------------
# Controller dispatch
# ---------------------------------------------------------------------------

def build_controller(method_dict: Dict[str, Any], cfg: NetCommConfig) -> ControllerProtocol:
    name = method_dict.get("policy", "per_packet_hmm")
    flags = {k: v for k, v in method_dict.items() if k not in ("policy", "note")}

    # lazy import — keeps this module compilable before Agent 1 lands
    from netcomm.routing import policies as P

    if name == "per_packet_hmm":
        return P.PerPacketHMMController(cfg, **flags)
    if name == "always_react":
        return P.AlwaysReact(cfg)
    if name == "always_predict":
        return P.AlwaysPredict(cfg)
    if name == "always_diversify":
        return P.AlwaysDiversify(cfg)
    if name == "oracle_regime":
        return P.OracleRegimeController(cfg)
    if name == "oracle":
        return P.OracleRouting(cfg)
    if name == "scalar_bfs_predictive":
        return P.ScalarBFSPredictive(cfg)
    if name == "gpsr":
        return P.ReactiveGPSR(cfg)
    if name == "glsr":
        return P.ReactiveGLSR(cfg)
    if name == "aodv":
        return P.ReactiveAODV(cfg)
    if name == "dsr":
        return P.ReactiveDSR(cfg)
    if name == "olsr":
        from netcomm.routing.olsr import OLSRPolicy
        return OLSRPolicy(cfg)
    if name == "polsr":
        from netcomm.routing.polsr import POLSRPolicy
        return POLSRPolicy(cfg)
    if name == "tgpsr":
        from netcomm.routing.tgpsr import TGPSRPolicy
        return TGPSRPolicy(cfg)
    if name == "p3":
        return P.P3Policy(cfg)
    if name == "car":
        return P.CARPolicy(cfg)
    if name == "learning_router":
        return P.LearningRouterPolicy(cfg)
    if name == "gnn_routing":
        return P.GNNRoutingPolicy(cfg)
    raise KeyError(f"Unknown policy name: {name!r}")


# ---------------------------------------------------------------------------
# Flow setup
# ---------------------------------------------------------------------------

def default_flows(cfg: NetCommConfig) -> List[Tuple[int, int]]:
    n = cfg.n_nodes
    return [(0, n - 1), (1, n - 2)] if n >= 4 else [(0, n - 1)]


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

def write_parquet(rows: List[Dict[str, Any]], table_name: str) -> Path:
    df = pd.DataFrame(rows)
    out_dir = RESULTS_DIR / table_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / "sweep.parquet"
    try:
        df.to_parquet(fname, index=False)
    except Exception:
        fname = fname.with_suffix(".csv")
        df.to_csv(fname, index=False)
    # also write a date-tagged snapshot for archival
    tagged = out_dir / f"{date.today().isoformat()}.parquet"
    try:
        df.to_parquet(tagged, index=False)
    except Exception:
        tagged = tagged.with_suffix(".csv")
        df.to_csv(tagged, index=False)
    return fname


SCENARIOS = ("open_field", "urban_canyon", "indoor_warehouse", "foliage", "long_corridor")
