import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax


# ---------------------------------------------------------------------------
# Model (JAX / Equinox)
# ---------------------------------------------------------------------------

# time invariant vector field directly in SONAR embedding space
class ODEFunc(eqx.Module):
    """Defines dx/dt = f(x). MLP operating directly in embedding space."""
    mlp: eqx.nn.MLP

    def __init__(self, d_embed: int, d_hidden: int, depth: int = 3, *, key: jax.Array):
        self.mlp = eqx.nn.MLP(
            in_size=d_embed,
            out_size=d_embed,
            width_size=d_hidden,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, _t: float, x: jax.Array, _args) -> jax.Array:
        return self.mlp(x)


class NeuralODE(eqx.Module):
    """Neural ODE operating directly in SONAR embedding space.

    No encoder/decoder — the ODE vector field is learned in the
    full embedding space. The initial condition x0 is the first
    observed embedding of a trace.
    """
    ode_func: ODEFunc

    def __init__(self, d_embed: int, d_ode_hidden: int, ode_depth: int = 3, *, key: jax.Array):
        self.ode_func = ODEFunc(d_embed, d_ode_hidden, depth=ode_depth, key=key)

    def solve(self, x0: jax.Array, ts: jax.Array) -> jax.Array:
        """Solve the ODE from x0, returning states at each timestamp in ts."""
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ode_func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=x0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5),
        )
        return solution.ys  # (T_max, d_embed)


# ---------------------------------------------------------------------------
# Loss: MSE between ODE trajectory and observed embeddings
# ---------------------------------------------------------------------------

def mse_single(
    model: NeuralODE,
    x: jax.Array,       # (T_max, d_embed)
    mask: jax.Array,     # (T_max,) bool
    ts: jax.Array,       # (T_max,) float
) -> jax.Array:
    """MSE loss for a single trajectory. x0 = first valid embedding."""
    x0 = x[0]  # first embedding is the initial condition
    x_hat = model.solve(x0, ts)  # (T_max, d_embed)

    # MSE over valid timesteps only
    sq_err = jnp.sum(mask[:, None] * (x_hat - x) ** 2) / jnp.sum(mask)
    return sq_err


def mse_batch(
    model: NeuralODE,
    padded: jax.Array,   # (B, T_max, d_embed)
    mask: jax.Array,     # (B, T_max) bool
    ts: jax.Array,       # (T_max,) float
) -> jax.Array:
    """Batch MSE loss via vmap."""
    def single(x, m):
        return mse_single(model, x, m, ts)

    per_sample = jax.vmap(single)(padded, mask)
    return jnp.mean(per_sample)
