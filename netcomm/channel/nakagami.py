import jax
import jax.numpy as jnp


@jax.jit
def nakagami_m_t(coh_time, m_0, T_0=1.0):
    return jnp.maximum(m_0 * jnp.minimum(coh_time / T_0, 1.0), 0.1)
