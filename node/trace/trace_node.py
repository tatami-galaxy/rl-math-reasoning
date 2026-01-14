import sys
sys.path.append('../../')
from dataclasses import dataclass, field

from utils import get_root_dir

from transformers import HfArgumentParser
from process_trace import download_layer_traces, segment_representations


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
    dataset_name: str = field(default="HuggingFaceH4/MATH-500")
    model_name: str = field(default="Qwen/Qwen3-4B-Thinking-2507")
    #trace_data: str = field(default="Ujan/qwen3-4b-thinking-math-trace")
    trace_layer: int = field(default=None)
    num_samples: int = field(default=500) # MATH500
    #trace_dir: str = field(default="/data/traces/")
    trace_dir: str = field(default="/data/Trace-Tensors/")
    segment_by: str = field(default="\n")

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
    print("WARNING : Make sure model name matches with trace data!")

    # download trace data if not already downloaded
    #trace_dir = download_layer_traces(root, config)

    # process data 
    segment_reps = segment_representations(root, config)

    # TODO : lower data dimensionality

    # init node
    

    

    