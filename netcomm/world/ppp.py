import jax
import jax.numpy as jnp
from .state import MAX_N, NodeState


@jax.jit
def sample_ppp(key, lambda_density, area_xy, z_range):
    x_min, x_max, y_min, y_max = area_xy
    area = (x_max - x_min) * (y_max - y_min)
    n_expected = lambda_density * area
    key, sk_n, sk_p = jax.random.split(key, 3)
    n_actual = jax.random.poisson(sk_n, n_expected)
    n_clip = jnp.minimum(n_actual, MAX_N)
    xs = jax.random.uniform(sk_p, (MAX_N,), minval=x_min, maxval=x_max)
    ys = jax.random.uniform(jax.random.fold_in(sk_p, 1), (MAX_N,),
                            minval=y_min, maxval=y_max)
    zs = jax.random.uniform(jax.random.fold_in(sk_p, 2), (MAX_N,),
                            minval=z_range[0], maxval=z_range[1])
    valid = jnp.arange(MAX_N) < n_clip
    return xs, ys, zs, valid


def fixed_n_nodes(key, n_nodes, area_xy, z_range):
    x_min, x_max, y_min, y_max = area_xy
    sk_x, sk_y, sk_z = jax.random.split(key, 3)
    xs = jnp.zeros(MAX_N).at[:n_nodes].set(
        jax.random.uniform(sk_x, (n_nodes,), minval=x_min, maxval=x_max))
    ys = jnp.zeros(MAX_N).at[:n_nodes].set(
        jax.random.uniform(sk_y, (n_nodes,), minval=y_min, maxval=y_max))
    zs = jnp.zeros(MAX_N).at[:n_nodes].set(
        jax.random.uniform(sk_z, (n_nodes,), minval=z_range[0], maxval=z_range[1]))
    valid = jnp.arange(MAX_N) < n_nodes
    return xs, ys, zs, valid


def add_ground_nodes(xs, ys, zs, valid, n_ground=2, anchor_xy=((0.0, 0.0), (50.0, 50.0))):
    for i in range(min(n_ground, len(anchor_xy))):
        ax, ay = anchor_xy[i]
        xs = xs.at[i].set(ax)
        ys = ys.at[i].set(ay)
        zs = zs.at[i].set(0.0)
    return xs, ys, zs, valid


def assign_classes(valid, n_ground=2):
    idx = jnp.arange(MAX_N)
    cls = jnp.where(idx < n_ground, 0, 1)
    return cls
