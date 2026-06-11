import math
from typing import Tuple, Optional, Dict

from netcomm.types import Packet, NetCommConfig, ActionInfo
from .costs import evaluate_costs
from .utility import (
    U_react, U_predict, U_diversify, U_drop,
    value_of_prediction, value_of_diversification,
)


def cache_key(src: int, dst: int, delta_p: float,
              quantum: float = 0.005) -> Tuple[int, int, int]:
    return (int(src), int(dst), int(math.ceil(max(delta_p, 0.0) / max(quantum, 1e-9))))


def pick_action(packet: Packet, belief_local, forecast, sinr, pi_up,
                adj, lcb, cfg: NetCommConfig,
                step_cache: Optional[Dict] = None,
                positions=None, log_buf: Optional[list] = None,
                disable_hmm: bool = False,
                disable_vop: bool = False,
                disable_vod: bool = False,
                disable_lcb: bool = False,
                disable_diversify: bool = False) -> ActionInfo:
    delta_p = float(packet.deadline - packet.t_gen)
    key = cache_key(int(packet.src), int(packet.dst), delta_p)
    if step_cache is not None and key in step_cache:
        return step_cache[key]

    if positions is None:
        import numpy as np
        positions = np.zeros((adj.shape[0], 3), dtype=float)

    # why: disable_hmm replaces the live belief with a flat prior so utilities
    # collapse to their no-belief Lagrangian. Matches "raw observations only".
    if disable_hmm:
        import numpy as np
        belief_local = np.full_like(np.asarray(belief_local, dtype=float), 0.25)

    # why: disable_lcb folds LCB back to pi_up so predict-flavored utilities
    # search the same graph as react. Matches "scalar BFS predictive".
    lcb_used = pi_up if disable_lcb else lcb

    net_state = {"n_hops_est": 3.0, "q_len": 0.0, "capacity": 256.0,
                 "n_fragments": cfg.n_fragments}

    c_r = evaluate_costs("react", packet, net_state, cfg)
    c_p = evaluate_costs("predict", packet, net_state, cfg)
    c_d = evaluate_costs("diversify", packet, net_state, cfg)

    u_r, path_r = U_react(packet, belief_local, pi_up, sinr, positions, adj, cfg, c_r)
    u_p, path_p = U_predict(packet, belief_local, lcb_used, adj, cfg, c_p)
    if disable_diversify:
        u_d, paths_d = float("-inf"), []
    else:
        u_d, paths_d = U_diversify(packet, belief_local, lcb_used, adj, cfg,
                                    c_d, cfg.k_paths)
    u_drop = U_drop(packet, belief_local)

    vop = value_of_prediction(u_p, u_r)
    vod = value_of_diversification(u_d, u_r, u_p)

    # why: disable_vop forces predict utility down so the controller never
    # escalates from react. disable_vod forces diversify utility down.
    if disable_vop:
        u_p = float("-inf")
    if disable_vod:
        u_d = float("-inf")

    utilities = [("react", u_r), ("predict", u_p),
                 ("diversify", u_d), ("drop", u_drop)]
    action, _u = max(utilities, key=lambda kv: kv[1])

    if action == "react":
        chosen_path, div_paths, s_pred = path_r, None, _path_value(path_r, pi_up)
    elif action == "predict":
        chosen_path, div_paths, s_pred = path_p, None, _path_value(path_p, lcb_used)
    elif action == "diversify":
        chosen_path, div_paths, s_pred = None, paths_d, _best_path_value(paths_d, lcb_used)
    else:
        chosen_path, div_paths, s_pred = None, None, 0.0

    info = ActionInfo(
        action=action,
        chosen_path=chosen_path,
        diversify_paths=div_paths,
        U_react=float(u_r),
        U_predict=float(u_p),
        U_diversify=float(u_d),
        U_drop=float(u_drop),
        VoP=float(vop),
        VoD=float(vod),
        S_pred=float(s_pred),
    )
    if step_cache is not None:
        step_cache[key] = info
    if log_buf is not None:
        log_buf.append(info)
    return info


def _path_value(path, values) -> float:
    if not path or len(path) < 2:
        return 0.0
    import numpy as np
    v = np.asarray(values)
    p = 1.0
    for u, w in zip(path[:-1], path[1:]):
        p *= float(v[u, w])
    return p


def _best_path_value(paths, values) -> float:
    if not paths:
        return 0.0
    return max(_path_value(p, values) for p in paths)
