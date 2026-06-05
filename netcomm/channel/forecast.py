import jax
import jax.numpy as jnp
from netcomm.types import ChannelForecast
from .doppler import doppler_spread, coherence_time, C_SPEED
from .pathloss import path_loss_gain
from .sinr import sinr_per_edge
from .los_3gpp import p_los_3gpp


def channel_forecast(node_traj, velocities, cfg, env_a=12.0, env_b=0.135):
    L, N, _ = node_traj.shape

    def per_step(positions):
        diff = positions[:, None, :] - positions[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1) + 1e-6
        sinr = sinr_per_edge(positions, cfg.p_tx, cfg.alpha_pl, cfg.beta_pl,
                             cfg.n0 * cfg.bandwidth)
        # Elevation per pair.
        horiz = jnp.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2) + 1e-6
        elev = jnp.rad2deg(jnp.arctan2(diff[..., 2], horiz))
        plos = p_los_3gpp(elev, env_a, env_b)
        return sinr, dist, plos

    sinr_t, dist_t, plos_t = jax.vmap(per_step)(node_traj)  # (L, N, N)

    # Relative velocity magnitude per pair (constant over horizon under CV).
    v_diff = velocities[:, None, :] - velocities[None, :, :]
    v_rel = jnp.linalg.norm(v_diff, axis=-1)  # (N, N)
    bd = doppler_spread(v_rel, cfg.f_c, cfg.cluster_kappa)  # (N, N)
    ct = coherence_time(bd)  # (N, N)
    bd_lnt = jnp.broadcast_to(bd, (L, N, N))
    ct_lnt = jnp.broadcast_to(ct, (L, N, N))

    # Transpose (L, N, N) -> (N, N, L).
    mean_sinr = jnp.transpose(sinr_t, (1, 2, 0))
    doppler = jnp.transpose(bd_lnt, (1, 2, 0))
    coh = jnp.transpose(ct_lnt, (1, 2, 0))
    plos = jnp.transpose(plos_t, (1, 2, 0))

    return ChannelForecast(mean_sinr=mean_sinr, doppler_spread=doppler,
                           coh_time=coh, p_los=plos)
