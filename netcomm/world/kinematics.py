import jax
import jax.numpy as jnp
from .state import NodeState


@jax.jit
def propagate_constant_velocity(pos, vel, dt):
    return pos + dt * vel


@jax.jit
def propagate_bounded_accel(pos, vel, acc, dt, a_max):
    acc_clipped = jnp.clip(acc, -a_max, a_max)
    new_vel = vel + dt * acc_clipped
    new_pos = pos + dt * vel + 0.5 * dt ** 2 * acc_clipped
    return new_pos, new_vel


def predict_trajectory(node_state: NodeState, n_horizon_steps: int, dt: float):
    def step(state, _):
        new_pos = state.pos + dt * state.vel
        new_state = state._replace(pos=new_pos)
        return new_state, new_pos
    _, traj = jax.lax.scan(step, node_state, None, length=n_horizon_steps)
    return traj
