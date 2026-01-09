import sys
sys.path.append('../../')
from dataclasses import dataclass, field

from utils import get_root_dir
from transformers import HfArgumentParser
from huggingface_hub import snapshot_download
from safetensors import safe_open

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

import diffrax
import equinox as eqx  

import matplotlib.pyplot as plt
import optax


def download_layer_traces(root, config):
    layer_pattern = "layer_"+str(config.trace_layer)+"/*"
    trace_dir = root+config.trace_dir+config.trace_data.split('/')[-1]
    snapshot_download(
        repo_id=config.trace_data,
        allow_patterns=layer_pattern,
        local_dir=trace_dir,
        token = config.hf_token,
    )


@dataclass
class TraceHyps:

    # seed
    seed: int = 42

    # trace data
    hf_token: str = field(default=None)
    trace_data: str = field(default="Ujan/qwen3-4b-thinking-math-trace")
    trace_layer: int = field(default=None)
    sample: bool = field(default=False)
    num_samples: int = field(default=10)
    trace_dir: str = field(default="/data/traces/")

    # neural ode
    node_depth: int = field(default=3)
    node_width: int = field(default=64)
    pid_rtol: float = field(default=1e-3)
    pid_atol: float = field(default=1e-6)


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


if __name__ == "__main__":

    root = get_root_dir()

    # get config
    parser = HfArgumentParser(TraceHyps)
    config = parser.parse_args_into_dataclasses()[0]
    if config.hf_token is None: raise ValueError("Pass in HF token")
    if config.trace_layer is None: raise ValueError("Pass in layer to get trace reps for")

    # download trace data if not already downloaded
    download_layer_traces(root, config)


    

    