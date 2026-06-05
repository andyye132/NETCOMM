import jax
import jax.numpy as jnp

C_SPEED = 3.0e8


@jax.jit
def doppler_spread(v_rel_mag, f_c, kappa):
    return f_c * v_rel_mag / C_SPEED * kappa


@jax.jit
def coherence_time(doppler_spread_val):
    return 1.0 / (4.0 * doppler_spread_val + 1e-12)
