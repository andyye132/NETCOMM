from typing import List, Optional, Tuple

import numpy as np

from netcomm.types import (
    NetCommConfig, NodeState, RegimeBelief, LCBSurvival, ChannelForecast,
    ControllerProtocol, ActionInfo, Packet,
)
from netcomm.routing.reactive import gpsr_route, glsr_route, aodv_forward, dsr_forward
from netcomm.routing.predictive import predictive_route
from netcomm.routing.p3 import p3_route
from netcomm.routing.car import car_route
from netcomm.routing.learning_router import learning_route
from netcomm.controller.decide import pick_action
from netcomm.regime.oracle import true_regime


class _BasePolicy:
    uses_regime: bool = False

    def __init__(self, cfg: NetCommConfig):
        self.cfg = cfg

    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        raise NotImplementedError


class ReactiveGPSR(_BasePolicy):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        sinr_np = np.asarray(sinr)
        return [gpsr_route(int(s), int(d), np.asarray(positions),
                           np.asarray(adj), sinr_np, self.cfg.gamma_th)
                for (s, d) in flows]


class ReactiveGLSR(_BasePolicy):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        return [glsr_route(int(s), int(d), np.asarray(positions),
                           np.asarray(adj), np.asarray(sinr),
                           np.asarray(pi_up), self.cfg.gamma_th)
                for (s, d) in flows]


class ReactiveAODV(_BasePolicy):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        return [aodv_forward(int(s), int(d), np.asarray(positions),
                             np.asarray(adj), np.asarray(sinr),
                             self.cfg.gamma_th)
                for (s, d) in flows]


class ReactiveDSR(_BasePolicy):
    def __init__(self, cfg: NetCommConfig):
        super().__init__(cfg)
        self._cache: dict = {}
        self._step = 0

    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        self._step += 1
        return [dsr_forward(int(s), int(d), np.asarray(positions),
                            np.asarray(adj), np.asarray(sinr),
                            self.cfg.gamma_th, cache=self._cache, step=self._step)
                for (s, d) in flows]


class PredictivePolicy(_BasePolicy):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        return [predictive_route(int(s), int(d), np.asarray(positions),
                                 np.asarray(adj), np.asarray(pi_up),
                                 self.cfg.pi_min)
                for (s, d) in flows]


class ScalarBFSPredictive(_BasePolicy):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        # why: ablation; scalar BFS on pi_up (no LCB).
        return [predictive_route(int(s), int(d), np.asarray(positions),
                                 np.asarray(adj), np.asarray(pi_up),
                                 self.cfg.pi_min)
                for (s, d) in flows]


class OracleRouting(_BasePolicy):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        return [predictive_route(int(s), int(d), np.asarray(positions),
                                 np.asarray(adj), np.asarray(pi_up), 0.0)
                for (s, d) in flows]


class P3Policy(_BasePolicy):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        vel = np.asarray(node_state.vel) if node_state is not None else np.zeros_like(positions)
        return [p3_route(int(s), int(d), np.asarray(positions), vel,
                         np.asarray(adj), np.asarray(pi_up), self.cfg.pi_min)
                for (s, d) in flows]


class CARPolicy(_BasePolicy):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        N = np.asarray(pi_up).shape[0]
        ctx = np.ones(N, dtype=np.float32)  # placeholder context score
        return [car_route(int(s), int(d), np.asarray(positions),
                          np.asarray(adj), np.asarray(pi_up), ctx,
                          self.cfg.pi_min)
                for (s, d) in flows]


class LearningRouterPolicy(_BasePolicy):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        return [learning_route(int(s), int(d), np.asarray(positions),
                               np.asarray(adj), np.asarray(pi_up),
                               np.asarray(sinr), self.cfg.gamma_th)
                for (s, d) in flows]


class GNNRoutingPolicy(_BasePolicy):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief=None, lcb=None, forecast=None):
        # why: stub — analytical predictive forward stands in for trained GAT.
        return [predictive_route(int(s), int(d), np.asarray(positions),
                                 np.asarray(adj), np.asarray(pi_up),
                                 self.cfg.pi_min)
                for (s, d) in flows]


# ----------------------------------------------------------------------------
# Per-packet belief-state controller (the new main policy)
# ----------------------------------------------------------------------------

class _PerPacketBase:
    uses_regime: bool = True

    def __init__(self, cfg: NetCommConfig):
        self.cfg = cfg

    def _wrap_flows_as_packets(self, flows, t: float = 0.0) -> List[Packet]:
        # why: the legacy route() contract is per-flow; we synthesize one Packet
        # per flow at the current time for the controller's argmax. Real runner
        # uses run_episode() which already has Packet objects.
        return [Packet(id=i, src=int(s), dst=int(d), t_gen=t,
                       deadline=t + self.cfg.T_AoI, priority=0, size=1)
                for i, (s, d) in enumerate(flows)]

    def _info_to_route(self, info: ActionInfo):
        if info.action == "drop":
            return None
        if info.action == "diversify":
            return info.diversify_paths or []
        return info.chosen_path or []


class PerPacketHMMController(_PerPacketBase):
    def __init__(self, cfg: NetCommConfig, *,
                 disable_hmm: bool = False,
                 disable_vop: bool = False,
                 disable_vod: bool = False,
                 disable_lcb: bool = False,
                 disable_diversify: bool = False,
                 two_state_hmm: bool = False,
                 ot_diversify: bool = False):
        super().__init__(cfg)
        self.flags = dict(
            disable_hmm=disable_hmm,
            disable_vop=disable_vop,
            disable_vod=disable_vod,
            disable_lcb=disable_lcb,
            disable_diversify=disable_diversify,
        )
        self.two_state_hmm = two_state_hmm
        self.ot_diversify = ot_diversify

    def _collapse_to_two_state(self, belief_local):
        # why: ablation - merge stable+predictable into "good" and volatile+blocked
        # into "bad", re-expand back to a 4-vector so downstream code is unchanged.
        b = np.asarray(belief_local, dtype=np.float64)
        if b.ndim == 1:
            good = b[0] + b[1]; bad = b[2] + b[3]
            return np.array([good / 2, good / 2, bad / 2, bad / 2])
        good = b[..., 0] + b[..., 1]
        bad = b[..., 2] + b[..., 3]
        return np.stack([good / 2, good / 2, bad / 2, bad / 2], axis=-1)

    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief: RegimeBelief, lcb: Optional[LCBSurvival] = None,
              forecast: Optional[ChannelForecast] = None):
        out = []
        cache: dict = {}
        pkts = self._wrap_flows_as_packets(flows)
        lcb_v = lcb.lcb if lcb is not None else pi_up
        for pkt in pkts:
            belief_local = regime_belief.b[int(pkt.src)]
            if self.two_state_hmm:
                belief_local = self._collapse_to_two_state(belief_local)
            info = pick_action(pkt, belief_local, forecast, sinr, pi_up,
                               np.asarray(adj), lcb_v, self.cfg,
                               step_cache=cache,
                               positions=np.asarray(positions),
                               **self.flags)
            out.append(self._info_to_route(info))
        return out


class AlwaysReact(PerPacketHMMController):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief, lcb=None, forecast=None):
        # why: ablation — force action=react regardless of utilities.
        return [gpsr_route(int(s), int(d), np.asarray(positions),
                           np.asarray(adj), np.asarray(sinr), self.cfg.gamma_th)
                for (s, d) in flows]


class AlwaysPredict(PerPacketHMMController):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief, lcb=None, forecast=None):
        from netcomm.controller.utility import _dijkstra_log_cost
        lcb_np = np.asarray(lcb.lcb if lcb is not None else pi_up)
        adj_np = np.asarray(adj)
        return [_dijkstra_log_cost(int(s), int(d), adj_np, lcb_np)
                for (s, d) in flows]


class AlwaysDiversify(PerPacketHMMController):
    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief, lcb=None, forecast=None):
        from netcomm.diversify.paths import k_disjoint_paths
        lcb_np = np.asarray(lcb.lcb if lcb is not None else pi_up)
        adj_np = np.asarray(adj)
        edge_cost = -np.log(np.clip(lcb_np, 1e-9, 1.0))
        return [k_disjoint_paths(adj_np, int(s), int(d),
                                 self.cfg.k_paths, edge_cost)
                for (s, d) in flows]


class OracleRegimeController(_PerPacketBase):
    uses_regime: bool = True

    def route(self, flows, pi_up, sinr, positions, adj, node_state,
              regime_belief, lcb=None, forecast=None):
        # why: replace HMM posterior with a one-hot at the true regime label.
        import jax.numpy as jnp
        from netcomm.types import RegimeBelief, N_REGIMES
        if forecast is None:
            return PerPacketHMMController(self.cfg).route(
                flows, pi_up, sinr, positions, adj, node_state,
                regime_belief, lcb, forecast)
        labels = true_regime(node_state, forecast, self.cfg.T_AoI)  # (N, N)
        N = int(labels.shape[0])
        one_hot = jnp.zeros((N, N, N_REGIMES))
        idx = jnp.eye(N_REGIMES)[labels]  # (N, N, R)
        oracle_belief = RegimeBelief(b=idx)
        controller = PerPacketHMMController(self.cfg)
        return controller.route(flows, pi_up, sinr, positions, adj, node_state,
                                oracle_belief, lcb, forecast)
