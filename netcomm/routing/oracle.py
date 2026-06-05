from .predictive import predictive_route


class OracleRouting:
    uses_wonham = False

    def __init__(self, cfg):
        self.cfg = cfg

    def route(self, flows, pi_up, sinr, positions, adj, node_state=None,
              wonham_state=None):
        # Simplification: canonical oracle would route on future-realized link outcomes,
        # but the runner does not expose them; we instead drop the pi_min gate so the
        # oracle searches the full max-product-reliability path over the forecast.
        return [predictive_route(s, d, positions, adj, pi_up, 0.0)
                for s, d in flows]
