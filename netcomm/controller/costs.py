from typing import Tuple

from netcomm.types import Packet, NetCommConfig


def evaluate_costs(action: str, packet: Packet, network_state: dict,
                   cfg: NetCommConfig) -> Tuple[float, float, float, float]:
    # why: simple cost proxies for the U(a) lagrangian. Bandwidth ~ packet size,
    # energy ~ size * hops, control overhead ~ 1 for actions that pay the
    # planner cost, queue ~ q_len / capacity. Diversify multiplies by fragment
    # fan-out to capture the redundancy tax.
    size = float(packet.size)
    n_hops_est = float(network_state.get("n_hops_est", 3.0))
    q_len = float(network_state.get("q_len", 0.0))
    capacity = max(float(network_state.get("capacity", 256.0)), 1.0)
    if action == "react":
        B = size
        E = size * n_hops_est
        C = 0.0
        Q = q_len / capacity
    elif action == "predict":
        B = size
        E = size * n_hops_est
        C = 1.0
        Q = q_len / capacity
    elif action == "diversify":
        fan = float(network_state.get("n_fragments", cfg.n_fragments))
        B = size * fan
        E = size * fan * n_hops_est
        C = 1.0
        Q = q_len / capacity
    elif action == "drop":
        B = 0.0
        E = 0.0
        C = 0.0
        Q = 0.0
    else:
        B = E = C = Q = 0.0
    return B, E, C, Q
