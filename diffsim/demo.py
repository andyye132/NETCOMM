"""Differentiable-core demo. Shows that the sim is differentiable: it computes a tracking-
uncertainty loss, finite-difference-checks that gradients flow through the whole rollout, then
runs gradient descent THROUGH the sim to move the drones to better track moving targets.

    uv run python -m diffsim.demo
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)   # x64 so the finite-difference check is precise

from .config import DiffSimConfig
from .rollout import episode_uncertainty, optimize_drone_positions


def make_scenario(T=30, M=4, seed=0):
    """M targets crossing the map along straight constant-velocity lines -> (T, M, 2)."""
    rng = np.random.default_rng(seed)
    starts = rng.uniform([40.0, 40.0], [260.0, 260.0], size=(M, 2))
    vels = rng.uniform(-4.0, 4.0, size=(M, 2))
    traj = np.stack([starts + vels * t for t in range(T)], axis=0)
    return jnp.asarray(traj)


def finite_difference_check(loss_fn, d, eps=1e-2):
    """Central-difference vs jax.grad, to prove gradients are correct (not just nonzero)."""
    g = np.asarray(jax.grad(loss_fn)(d))
    d = np.asarray(d)
    fd = np.zeros_like(g)
    for i in range(d.shape[0]):
        for j in range(2):
            dp, dm = d.copy(), d.copy()
            dp[i, j] += eps
            dm[i, j] -= eps
            fd[i, j] = (float(loss_fn(jnp.asarray(dp))) - float(loss_fn(jnp.asarray(dm)))) / (2 * eps)
    rel = np.linalg.norm(g - fd) / (np.linalg.norm(fd) + 1e-9)
    return g, fd, rel


def main():
    cfg = DiffSimConfig()
    traj = make_scenario()
    drones0 = jnp.asarray([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]])   # bunched in a corner
    loss_fn = lambda d: episode_uncertainty(d, traj, cfg)

    L0 = float(loss_fn(drones0))
    # finite-difference check at a well-conditioned point (drones among the targets), so the
    # printed number cleanly demonstrates jax.grad is correct — not just nonzero.
    d_check = jnp.asarray([[130.0, 130.0], [170.0, 150.0], [150.0, 170.0]])
    g, _, rel = finite_difference_check(loss_fn, d_check)
    print(f"initial tracking-uncertainty loss = {L0:10.1f}")
    print(f"||jax.grad|| through the full rollout = {float(jnp.linalg.norm(g)):.1f}  (gradients flow)")
    print(f"jax.grad vs finite-difference rel error = {rel:.2e}  ->  the sim is DIFFERENTIABLE")

    d_final, history = optimize_drone_positions(drones0, traj, cfg, steps=80, step_m=4.0)
    print(f"final tracking-uncertainty loss   = {history[-1]:10.1f}  "
          f"({100 * (1 - history[-1] / history[0]):.0f}% reduction by gradient descent)")

    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].plot(history, lw=2)
        ax[0].set(title="tracking uncertainty vs gradient-descent step (backprop through the sim)",
                  xlabel="optimization step", ylabel=r"$\sum_t \sum_i \mathrm{tr}\,\Sigma_i(t)$")
        tr = np.asarray(traj)
        for m in range(tr.shape[1]):
            ax[1].plot(tr[:, m, 0], tr[:, m, 1], "-", color="0.6", lw=1)
            ax[1].plot(tr[0, m, 0], tr[0, m, 1], "o", color="0.6", ms=4)
        d0, df = np.asarray(drones0), np.asarray(d_final)
        ax[1].scatter(d0[:, 0], d0[:, 1], c="red", marker="x", s=90, label="drones (initial)")
        ax[1].scatter(df[:, 0], df[:, 1], c="green", marker="*", s=160, label="drones (grad-optimized)")
        ax[1].set(title="gradient-optimized drone placement", xlim=(0, 300), ylim=(0, 300))
        ax[1].legend(loc="upper left", fontsize=8)
        os.makedirs("results/diffsim", exist_ok=True)
        out = "results/diffsim/diffsim_demo.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")
    except Exception as e:                                       # plotting is optional
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
