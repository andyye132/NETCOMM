from typing import List, Tuple, Optional, Dict

import jax
import jax.numpy as jnp
import numpy as np

from netcomm.types import (
    NodeState, NetCommConfig, RegimeBelief, ActionInfo, MAX_N,
)
from netcomm.world.ppp import (
    sample_ppp, fixed_n_nodes, add_ground_nodes, assign_classes,
)
from netcomm.world.kinematics import predict_trajectory
from netcomm.world.environment import get_env_params
from netcomm.world.topology import build_graph
from netcomm.channel.forecast import channel_forecast
from netcomm.channel.linkup import linkup_by_deadline
from netcomm.channel.sinr import sinr_per_edge
from netcomm.lcb.survival import lcb_link_survival
from netcomm.regime.filter import init_belief, build_link_transition, regime_step
from netcomm.regime.observations import collect_observations, observation_likelihood
from netcomm.regime.transitions import (
    doppler_severity_from_coh, blockage_rate_from_missed,
)
from netcomm.packets.queue import PriorityQueue
from netcomm.packets.generator import PoissonGenerator
from netcomm.controller.decide import pick_action
from netcomm.aoi.tracker import AoITracker
from netcomm.metrics.delivery import delivery_probability
from netcomm.metrics.aoi_metrics import mean_aoi, percentile_latency
from netcomm.metrics.runtime import RuntimeRecorder


class NetCommWorld:
    def __init__(self, cfg: NetCommConfig, key):
        self.cfg = cfg
        env = get_env_params(cfg.env)
        self.env_a = env["env_a"]
        self.env_b = env["env_b"]
        sk_init, sk_v = jax.random.split(key)
        if cfg.n_nodes > 0:
            xs, ys, zs, valid = fixed_n_nodes(sk_init, cfg.n_nodes,
                                              cfg.area_xy, cfg.z_range)
        else:
            xs, ys, zs, valid = sample_ppp(sk_init, cfg.lambda_density,
                                           cfg.area_xy, cfg.z_range)
        xs, ys, zs, valid = add_ground_nodes(xs, ys, zs, valid)
        cls = assign_classes(valid)
        sk_vx, sk_vy = jax.random.split(sk_v)
        v_max = 10.0
        vx = jnp.where(cls == 0, 0.0,
                       jax.random.uniform(sk_vx, (MAX_N,), minval=-v_max, maxval=v_max))
        vy = jnp.where(cls == 0, 0.0,
                       jax.random.uniform(sk_vy, (MAX_N,), minval=-v_max, maxval=v_max))
        self.state = NodeState(
            pos=jnp.stack([xs, ys, zs], axis=-1),
            vel=jnp.stack([vx, vy, jnp.zeros(MAX_N)], axis=-1),
            heading=jnp.zeros((MAX_N, 3)),
            class_id=cls,
        )
        self.valid = valid
        self.last_outcomes = None

    def advance(self, dt):
        cfg = self.cfg
        new_pos = self.state.pos + dt * self.state.vel
        xmn, xmx, ymn, ymx = cfg.area_xy
        nx = jnp.where(new_pos[:, 0] < xmn, xmn + (xmn - new_pos[:, 0]),
                       jnp.where(new_pos[:, 0] > xmx, xmx - (new_pos[:, 0] - xmx),
                                 new_pos[:, 0]))
        ny = jnp.where(new_pos[:, 1] < ymn, ymn + (ymn - new_pos[:, 1]),
                       jnp.where(new_pos[:, 1] > ymx, ymx - (new_pos[:, 1] - ymx),
                                 new_pos[:, 1]))
        self.state = self.state._replace(
            pos=jnp.stack([nx, ny, new_pos[:, 2]], axis=-1),
        )


def sample_per_packet_delivery(forwarded, pi_up, lcb, key) -> List[dict]:
    # why: per-packet Bernoulli-sample delivery along chosen path(s). For
    # diversify we treat the bundle as delivered if k-of-n path survivals hit
    # (here we use the union of independent paths approximation).
    pi_np = np.asarray(pi_up)
    lcb_np = np.asarray(lcb.lcb if hasattr(lcb, "lcb") else lcb)
    outs = []
    for (pkt, info) in forwarded:
        action = info.action
        sk = jax.random.fold_in(key, int(pkt.id) % (2 ** 30))
        u = float(jax.random.uniform(sk))
        delivered = 0
        n_hops = 0
        if action == "react" and info.chosen_path:
            path = info.chosen_path
            n_hops = max(len(path) - 1, 0)
            p = 1.0
            for a, b in zip(path[:-1], path[1:]):
                p *= float(pi_np[a, b])
            delivered = int(u < p)
        elif action == "predict" and info.chosen_path:
            path = info.chosen_path
            n_hops = max(len(path) - 1, 0)
            p = 1.0
            for a, b in zip(path[:-1], path[1:]):
                p *= float(pi_np[a, b])
            delivered = int(u < p)
        elif action == "diversify" and info.diversify_paths:
            survs = []
            for path in info.diversify_paths:
                if len(path) < 2:
                    continue
                p = 1.0
                for a, b in zip(path[:-1], path[1:]):
                    p *= float(pi_np[a, b])
                survs.append(p)
                n_hops = max(n_hops, len(path) - 1)
            # P(at least one survives) under independence
            p_any = 1.0
            for s in survs:
                p_any *= (1.0 - s)
            p_any = 1.0 - p_any
            delivered = int(u < p_any)
        outs.append({
            "id": int(pkt.id),
            "src": int(pkt.src),
            "dst": int(pkt.dst),
            "action": action,
            "delivered": delivered,
            "n_hops": int(n_hops),
            "info": info,
            "t_gen": float(pkt.t_gen),
        })
    return outs


def update_belief_from_outcomes(belief: RegimeBelief,
                                outcomes: List[dict],
                                cfg: NetCommConfig) -> RegimeBelief:
    # why: a single Bayes-style sharpening on observed src->next_hop links.
    # ACK -> push mass toward stable/predictable; NACK -> push toward
    # volatile/blocked. We use a small additive bump and renormalize.
    if not outcomes:
        return belief
    b = np.array(belief.b)
    eps = 0.05
    for o in outcomes:
        info = o["info"]
        path = None
        if info.action in ("react", "predict") and info.chosen_path:
            path = info.chosen_path
        elif info.action == "diversify" and info.diversify_paths:
            path = info.diversify_paths[0]
        if not path or len(path) < 2:
            continue
        s, nh = int(path[0]), int(path[1])
        if o["delivered"]:
            bump = np.array([eps, eps * 0.5, -eps * 0.5, -eps], dtype=b.dtype)
        else:
            bump = np.array([-eps, -eps * 0.5, eps * 0.5, eps], dtype=b.dtype)
        row = b[s, nh] + bump
        row = np.clip(row, 1e-6, None)
        row = row / row.sum()
        b[s, nh] = row
    return RegimeBelief(b=jnp.asarray(b))


def run_episode(cfg: NetCommConfig, controller, flows: List[Tuple[int, int]],
                n_steps: int, key, record_snapshots: bool = False) -> Dict:
    key, sk = jax.random.split(key)
    world = NetCommWorld(cfg, sk)
    N = int(world.state.pos.shape[0])
    belief = init_belief(N)
    queue = PriorityQueue()
    gen = PoissonGenerator(cfg.pkt_rate_per_class, cfg.dt, T_AoI=cfg.T_AoI)
    rec = RuntimeRecorder()
    aoi = AoITracker()

    all_deliveries: List[int] = []
    all_latencies: List[float] = []
    action_log: List[str] = []
    vop_log: List[float] = []
    vod_log: List[float] = []
    info_log: List[ActionInfo] = []
    snapshots = [] if record_snapshots else None

    n_dropped_stale = 0
    n_packets_total = 0
    next_id = 0

    for t_step in range(n_steps):
        rec.lap("pre")
        t = t_step * cfg.dt
        world.advance(cfg.dt)
        traj = predict_trajectory(world.state, cfg.n_horizon_steps, cfg.dt)
        forecast = channel_forecast(traj, world.state.vel, cfg,
                                    env_a=world.env_a, env_b=world.env_b)
        pi_up = linkup_by_deadline(forecast, cfg.gamma_th, cfg.r_rng,
                                   cfg.m_0, T_0=cfg.T_0)
        sinr = sinr_per_edge(world.state.pos, cfg.p_tx, cfg.alpha_pl,
                             cfg.beta_pl, cfg.n0 * cfg.bandwidth)
        adj = build_graph(world.state.pos, cfg.r_rng, world.valid)
        lcb = lcb_link_survival(forecast, cfg.T_AoI, z_delta=cfg.z_delta)
        rec.lap("forecast")

        # Build link transition + emission and update regime belief.
        coh_mean = jnp.mean(forecast.coh_time, axis=-1)
        doppler_sev = doppler_severity_from_coh(coh_mean, cfg.T_AoI)
        # No beacon-missed signal yet; supply zeros.
        block_rate = jnp.zeros_like(doppler_sev)
        tm = build_link_transition(cfg.dt, doppler_sev, block_rate)
        obs = collect_observations(world, world.last_outcomes, None, forecast)
        emiss = observation_likelihood(obs, None)
        belief = regime_step(belief, emiss, tm)
        rec.lap("belief")

        # Generate new packets for each flow and admit to queue.
        new_pkts, next_id = gen.generate(t, cfg.dt, list(flows), next_id)
        n_packets_total += len(new_pkts)
        queue.extend(new_pkts)

        step_cache: Dict = {}
        forwarded: List[Tuple] = []
        for pkt in queue.drain_for_step():
            if pkt.deadline <= t:
                n_dropped_stale += 1
                continue
            belief_local = belief.b[int(pkt.src)]  # (N, 4) outgoing rows
            info = pick_action(pkt, belief_local, forecast, sinr, pi_up,
                               np.asarray(adj), lcb.lcb, cfg,
                               step_cache=step_cache,
                               positions=np.asarray(world.state.pos),
                               log_buf=None)
            action_log.append(info.action)
            vop_log.append(info.VoP)
            vod_log.append(info.VoD)
            info_log.append(info)
            if info.action == "drop":
                continue
            forwarded.append((pkt, info))
        rec.lap("decide")

        key, sk_step = jax.random.split(key)
        outcomes = sample_per_packet_delivery(forwarded, pi_up, lcb, sk_step)
        for o in outcomes:
            all_deliveries.append(o["delivered"])
            lat_ms = o["n_hops"] * cfg.dt * 1e3
            if not o["delivered"]:
                lat_ms = cfg.T_AoI * 1e3
            all_latencies.append(lat_ms)
            aoi.update(lat_ms)

        belief = update_belief_from_outcomes(belief, outcomes, cfg)
        rec.lap("deliver")

        # Track an outcomes summary for next-step observation.
        if outcomes:
            mean_succ = float(np.mean([o["delivered"] for o in outcomes]))
            world.last_outcomes = jnp.full((N, N), mean_succ, dtype=jnp.float32)

        if record_snapshots:
            snapshots.append({
                "t": t_step,
                "positions": np.asarray(world.state.pos),
                "valid": np.asarray(world.valid, dtype=bool),
                "class_id": np.asarray(world.state.class_id),
                "pi_up": np.asarray(pi_up),
                "lcb": np.asarray(lcb.lcb),
                "routes": [o["info"].chosen_path
                           if o["info"].action in ("react", "predict")
                           else (o["info"].diversify_paths or [])
                           for o in outcomes],
                "actions": [o["action"] for o in outcomes],
                "flows": list(flows),
                "regime_belief": np.asarray(belief.b),
                "delivery": (float(np.mean([o["delivered"] for o in outcomes]))
                             if outcomes else 0.0),
            })

    # Mode occupancy as fractions.
    if action_log:
        from collections import Counter
        counts = Counter(action_log)
        total = float(len(action_log))
        mode_occupancy = {k: counts.get(k, 0) / total
                          for k in ("react", "predict", "diversify", "drop")}
    else:
        mode_occupancy = {k: 0.0 for k in ("react", "predict", "diversify", "drop")}

    result = {
        "delivery_probability": delivery_probability(all_deliveries),
        "mean_aoi": mean_aoi(all_latencies),
        "p99_latency": percentile_latency(all_latencies, 0.99),
        "mode_occupancy": mode_occupancy,
        "vop_log": vop_log,
        "vod_log": vod_log,
        "n_packets": int(n_packets_total),
        "n_dropped_stale": int(n_dropped_stale),
        "runtime_per_step": rec.summary(),
    }
    if record_snapshots:
        result["snapshots"] = snapshots
    return result
