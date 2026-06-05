from .predictive import predictive_route


class GNNRoutingPolicy:

    uses_wonham = False

    def __init__(self, cfg):
        self.cfg = cfg

    def route(self, flows, pi_up, sinr, positions, adj, node_state=None,
              wonham_state=None):
        out = []
        for s, d in flows:
            out.append(predictive_route(s, d, positions, adj, pi_up,
                                         self.cfg.pi_min))
        return out
