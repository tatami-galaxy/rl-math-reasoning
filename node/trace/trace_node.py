import sys
sys.path.append('../../')
from dataclasses import dataclass, field

from utils import get_root_dir

from transformers import HfArgumentParser
from process_trace import download_layer_traces
from huggingface_hub import snapshot_download
from safetensors import safe_open

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import diffrax
import equinox as eqx 
from node import NeuralODE
import optax

import matplotlib.pyplot as plt


@dataclass
class TraceHyps:

    # seed
    seed: int = 42

    # trace data
    hf_token: str = field(default=None)
    trace_data: str = field(default="Ujan/qwen3-4b-thinking-math-trace")
    trace_layer: int = field(default=None)
    num_samples: int = field(default=500) # MATH500
    trace_dir: str = field(default="/data/traces/")

    # neural ode
    node_depth: int = field(default=3)
    node_width: int = field(default=64)
    pid_rtol: float = field(default=1e-3)
    pid_atol: float = field(default=1e-6)


if __name__ == "__main__":

    root = get_root_dir()

    # get config
    parser = HfArgumentParser(TraceHyps)
    config = parser.parse_args_into_dataclasses()[0]
    if config.hf_token is None: raise ValueError("Pass in HF token")
    if config.trace_layer is None: raise ValueError("Pass in layer to get trace reps for")

    # download trace data if not already downloaded
    trace_dir = download_layer_traces(root, config)

    # process data


    layer_data_dir = trace_dir+"/layer_"+str(config.trace_layer)
    with safe_open(layer_data_dir+"/1.safetensors", framework="jax", device="cpu") as f:
        metadata = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        print(tensors['layer_trace'].shape)
        print(type(tensors['layer_trace']))


    

    