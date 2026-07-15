"""Tests that the differentiable core is genuinely differentiable: gradients flow through the
whole rollout, match finite differences, and gradient descent reduces tracking uncertainty."""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from diffsim.config import DiffSimConfig
from diffsim.rollout import episode_uncertainty, optimize_drone_positions

cfg = DiffSimConfig()


def _traj(T=12, M=3, seed=1):
    rng = np.random.default_rng(seed)
    s = rng.uniform([50.0, 50.0], [250.0, 250.0], size=(M, 2))
    v = rng.uniform(-3.0, 3.0, size=(M, 2))
    return jnp.asarray(np.stack([s + v * t for t in range(T)], axis=0))


def test_loss_finite_and_jittable():
    traj = _traj()
    d = jnp.array([[100.0, 100.0], [150.0, 150.0]])
    L = episode_uncertainty(d, traj, cfg)
    assert jnp.isfinite(L) and float(L) > 0
    Lj = jax.jit(lambda x: episode_uncertainty(x, traj, cfg))(d)
    assert abs(float(L) - float(Lj)) < 1e-6


def test_gradients_flow_and_match_finite_difference():
    traj = _traj()
    d = jnp.array([[80.0, 90.0], [170.0, 160.0]])
    fn = lambda x: episode_uncertainty(x, traj, cfg)
    g = np.asarray(jax.grad(fn)(d))
    assert np.all(np.isfinite(g)) and np.linalg.norm(g) > 0           # gradients actually flow
    eps = 1e-3
    fd = np.zeros_like(g)
    dn = np.asarray(d)
    for i in range(dn.shape[0]):
        for j in range(2):
            dp, dm = dn.copy(), dn.copy()
            dp[i, j] += eps
            dm[i, j] -= eps
            fd[i, j] = (float(fn(jnp.asarray(dp))) - float(fn(jnp.asarray(dm)))) / (2 * eps)
    assert np.linalg.norm(g - fd) / np.linalg.norm(fd) < 1e-3          # and they are CORRECT


def test_optimization_reduces_uncertainty():
    traj = _traj()
    d0 = jnp.array([[20.0, 20.0], [30.0, 20.0], [20.0, 30.0]])        # bunched in a corner
    _, history = optimize_drone_positions(d0, traj, cfg, steps=60, step_m=4.0)
    assert history[-1] < 0.7 * history[0]                             # backprop tracks better


def test_grad_steers_drone_toward_target():
    # one drone just outside a static target's footprint; -grad must push it toward the target (+x)
    traj = jnp.broadcast_to(jnp.array([[150.0, 150.0]]), (8, 1, 2))
    d = jnp.array([[125.0, 150.0]])
    g = np.asarray(jax.grad(lambda x: episode_uncertainty(x, traj, cfg))(d))
    assert -g[0, 0] > abs(g[0, 1])                                    # descent points toward target
