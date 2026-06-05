import jax
import jax.numpy as jnp


@jax.jit
def path_loss_gain(dist, alpha, beta):
    return beta * jnp.power(dist + 1e-6, -alpha)
