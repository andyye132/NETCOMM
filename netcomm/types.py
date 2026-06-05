from typing import NamedTuple, Tuple, List, Optional, Dict, Any, Union
import jax.numpy as jnp
import numpy as np

MAX_N = 128
REGIME_NAMES = ("stable", "predictable", "volatile", "blocked")
ACTION_NAMES = ("react", "predict", "diversify", "drop")
N_REGIMES = 4
N_ACTIONS = 4


class NodeState(NamedTuple):
    pos: jnp.ndarray
    vel: jnp.ndarray
    heading: jnp.ndarray
    class_id: jnp.ndarray


class ChannelForecast(NamedTuple):
    mean_sinr: jnp.ndarray
    doppler_spread: jnp.ndarray
    coh_time: jnp.ndarray
    p_los: jnp.ndarray


class LinkUpProb(NamedTuple):
    pi_up: jnp.ndarray


class RegimeBelief(NamedTuple):
    b: jnp.ndarray


class LCBSurvival(NamedTuple):
    mean: jnp.ndarray
    std: jnp.ndarray
    lcb: jnp.ndarray


class Packet(NamedTuple):
    id: int
    src: int
    dst: int
    t_gen: float
    deadline: float
    priority: int
    size: int


class ActionInfo(NamedTuple):
    action: str
    chosen_path: Optional[List[int]]
    diversify_paths: Optional[List[List[int]]]
    U_react: float
    U_predict: float
    U_diversify: float
    U_drop: float
    VoP: float
    VoD: float
    S_pred: float


class NetCommConfig(NamedTuple):
    f_c: float
    bandwidth: float
    p_tx: float
    n0: float
    gamma_th: float
    r_rng: float
    T_AoI: float
    pi_min: float
    alpha_pl: float
    beta_pl: float
    m_0: float
    T_0: float
    cluster_kappa: float
    env: str
    lambda_density: float
    n_horizon_steps: int
    dt: float
    area_xy: Tuple[float, float, float, float]
    z_range: Tuple[float, float]
    n_nodes: int
    lambda_B: float = 0.01
    lambda_E: float = 0.01
    lambda_C: float = 0.005
    lambda_Q: float = 0.005
    z_delta: float = 1.282
    k_paths: int = 3
    n_fragments: int = 4
    k_decode: int = 2
    priority_classes: int = 3
    pkt_rate_per_class: Tuple[float, ...] = (10.0, 30.0, 5.0)
    compute_budget_per_step: int = 256


PathLike = List[int]
RouteResult = Union[PathLike, List[PathLike], None]


class ControllerProtocol:
    uses_regime: bool = True

    def route(
        self,
        flows: List[Tuple[int, int]],
        pi_up: jnp.ndarray,
        sinr: jnp.ndarray,
        positions: np.ndarray,
        adj: np.ndarray,
        node_state: NodeState,
        regime_belief: RegimeBelief,
        lcb: Optional[LCBSurvival] = None,
        forecast: Optional[ChannelForecast] = None,
    ) -> List[RouteResult]:
        raise NotImplementedError
