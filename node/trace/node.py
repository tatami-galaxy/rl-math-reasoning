import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

import diffrax
import equinox as eqx  


class Func(eqx.Module):
    out_scale: jax.Array
    mlp: eqx.nn.MLP

    def __init__(self, data_size, config, *, key, **kwargs):
        super().__init__(**kwargs)
        self.out_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=config.node_width,
            depth=config.node_depth,
            activation=jnn.softplus,
            final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        # standard practice is often to use `learnt_scalar * tanh(MLP(...))` for the vector field.
        return self.out_scale * self.mlp(y)


class NeuralODE(eqx.Module):
    func: Func
    rtol: float
    atol: float

    # * means everything after it must be passed by keyword
    def __init__(self, data_size, config, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, config.node_width, config.node_depth, key=key)
        self.rtol = config.pid_rtol
        self.atol = config.pid_atol


    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol, atol=self.atol
            ),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys