import jax.numpy as jnp

from netcomm.types import ChannelForecast, LCBSurvival


def lcb_link_survival(forecast: ChannelForecast, deadline: float,
                      z_delta: float = 1.282) -> LCBSurvival:
    # why: bootstrap-free first-moment LCB. Use the forecast horizon slices as a
    # natural ensemble: per-link survival sample s_l = sigmoid(SINR_l) over l in [0, L).
    # mean ± z * std gives a lower-confidence survival score.
    sinr = forecast.mean_sinr  # (N, N, L)
    p_step = sinr / (sinr + 1.0)  # logistic-style mapping to (0, 1)
    p_step = jnp.clip(p_step, 0.0, 1.0)
    # Treat horizon slices as bootstrap samples of link survival.
    mean = jnp.mean(p_step, axis=-1)
    std = jnp.std(p_step, axis=-1)
    lcb = jnp.clip(mean - z_delta * std, 0.0, 1.0)
    return LCBSurvival(mean=mean, std=std, lcb=lcb)


def survival_edge_costs(lcb: jnp.ndarray) -> jnp.ndarray:
    return -jnp.log(jnp.clip(lcb, 1e-9, 1.0))
