import jax.numpy as jnp


def true_regime(world_state, channel_forecast, T_AoI: float) -> jnp.ndarray:
    # why: 4-class hard-threshold oracle used as upper bound for Test 5/10.
    #   stable      : coh > 2*T_AoI and p_los > 0.9
    #   predictable : coh in (0.5, 2.0]*T_AoI
    #   volatile    : coh < 0.5 * T_AoI and p_los > 0.3
    #   blocked     : p_los < 0.3
    coh = jnp.mean(channel_forecast.coh_time, axis=-1)  # (N, N)
    plos = jnp.mean(channel_forecast.p_los, axis=-1)  # (N, N)

    # why: low plos alone is just attenuation. Blocked requires BOTH heavy
    # NLoS AND fast decorrelation.
    is_blocked = (plos < 0.15) & (coh < 0.5 * T_AoI)
    is_volatile = (~is_blocked) & (coh < 0.5 * T_AoI)
    is_stable = (~is_blocked) & (coh > 2.0 * T_AoI) & (plos > 0.7)
    is_pred = (~is_blocked) & (~is_volatile) & (~is_stable)

    label = (jnp.where(is_stable, 0, 0)
             + jnp.where(is_pred, 1, 0)
             + jnp.where(is_volatile, 2, 0)
             + jnp.where(is_blocked, 3, 0))
    return label.astype(jnp.int32)
