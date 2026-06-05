import jax.numpy as jnp


def doppler_severity_from_coh(coh_time: jnp.ndarray, T_AoI: float) -> jnp.ndarray:
    # why: severity 1.0 when coherence << deadline, 0.0 when coherence >> deadline.
    ratio = jnp.clip(T_AoI / jnp.clip(coh_time, 1e-6, None), 0.0, 10.0)
    return jnp.clip(ratio / (1.0 + ratio), 0.0, 1.0)


def blockage_rate_from_missed(missed_per_link: jnp.ndarray,
                              beacon_interval: float) -> jnp.ndarray:
    # why: fraction of expected beacons missed in the recent window.
    # missed_per_link is a count; normalize by an expected count over the window.
    expected = max(beacon_interval, 1.0)
    return jnp.clip(missed_per_link / expected, 0.0, 1.0)
