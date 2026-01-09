import sys
sys.path.append('../../')
from dataclasses import dataclass, field

from utils import get_root_dir
from transformers import HfArgumentParser
from huggingface_hub import snapshot_download
from safetensors import safe_open

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

import diffrax
import equinox as eqx  

import matplotlib.pyplot as plt
import optax


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


def get_layer_traces():
    pass


if __name__ == "__main__":

    root = get_root_dir()

    # get config
    parser = HfArgumentParser(TraceHyps)
    config = parser.parse_args_into_dataclasses()[0]
    if config.hf_token is None: raise ValueError("Pass in HF token")
    if config.trace_layer is None: raise ValueError("Pass in layer to get trace reps for")

    # download trace data if not already downloaded
    layer_pattern = "layer_"+str(config.trace_layer)+"/*"
    trace_dir = root+config.trace_dir+config.trace_data.split('/')[-1]
    snapshot_download(
        repo_id=config.trace_data,
        allow_patterns=layer_pattern,
        local_dir=trace_dir,
        token = config.hf_token,
    )

    