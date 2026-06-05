import jax
import jax.numpy as jnp


@jax.jit
def p_los_3gpp(elevation_deg, env_a, env_b):
    val = 1.0 + env_a * jnp.exp(-env_b * (elevation_deg - env_a))
    return 1.0 / val


@jax.jit
def elevation_deg(pos_i, pos_j):
    diff = pos_j - pos_i
    horiz = jnp.sqrt(diff[0] ** 2 + diff[1] ** 2) + 1e-6
    return jnp.rad2deg(jnp.arctan2(diff[2], horiz))
