import json

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

import diffrax
import equinox as eqx  


def save_node(filename, hyperparams, model):
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, model)


def load_node(filename):
    with open(filename, "rb") as f:

        hyperparams = json.loads(f.readline().decode())
        seed = hyperparams["seed"]
        key = jr.PRNGKey(seed)
        model_key, _ = jr.split(key, 2)
        node_width = hyperparams["node_width"]
        node_depth = hyperparams["node_depth"]
        data_size = hyperparams["data_size"]
        pid_rtol = hyperparams["pid_rtol"]
        pid_atol = hyperparams["pid_atol"]

        model = NeuralODE(
            node_width, node_depth, data_size, 
            pid_rtol, pid_atol, key=model_key
        )
        
        return eqx.tree_deserialise_leaves(f, model)


class VF(eqx.Module):
    out_scale: jax.Array
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.out_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jax.nn.tanh,
            key=key,
        )


    def __call__(self, t, y, args):
        # standard practice is often to use `learnt_scalar * tanh(MLP(...))` for the vector field
        # time independent vector field
        return self.out_scale * self.mlp(y)
    

    def div(self, t, x, args):
        jac = jax.jacfwd(lambda x: self(t, x, args))(x)
        return jnp.trace(jac)


class NeuralODE(eqx.Module):
    vf: VF
    rtol: float
    atol: float

    # * means everything after it must be passed by keyword
    def __init__(
            self, 
            node_width, node_depth, data_size, 
            pid_rtol, pid_atol,
            *, key, **kwargs
        ):
        super().__init__(**kwargs)
        self.vf = VF(data_size, node_width, node_depth, key=key)
        self.rtol = pid_rtol
        self.atol = pid_atol


    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.vf),
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