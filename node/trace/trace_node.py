import sys
sys.path.append('../../')
from dataclasses import dataclass, field

from utils import get_root_dir

from transformers import HfArgumentParser
from process_trace import segment_representations, load_segment_representations


import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
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
    trace_dir: str = field(default="/data/Trace-Tensors")
    segment_by: str = field(default="\n")
    resegment: bool = field(default=False)

    # neural ode
    node_depth: int = field(default=3)
    node_width: int = field(default=64)
    pid_rtol: float = field(default=1e-3)
    pid_atol: float = field(default=1e-6)
    total_steps: int = field(default=1000)
    step_strategy: list[float] = field(default_factory=lambda: [0.5, 0.5])
    length_strategy: list[float] = field(default_factory=lambda: [0.1, 1])

    # optimizer
    lr: float = field(default=1e-3)


@eqx.filter_value_and_grad
def grad_loss(model, ti, yi):
    y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
    return jnp.mean((yi - y_pred) ** 2)


@eqx.filter_jit
def make_step(ti, yi, model, opt_state):
    loss, grads = grad_loss(model, ti, yi)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train(config, model, optim):
    # Up until step 500 we train on only the first 10% of each time series.
    # This is a standard trick to avoid getting caught in a local minimum.
    for step_frac, length_frac in zip(config.step_strategy, config.length_strategy):

        # num steps according to step strategy
        steps = int(config.total_steps * step_frac)

        # TODO : this has to be set per example since examples are of different lengths
        # FIX : length = int(length_of_this_example * length_frac)

        # optimizer state
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        # time
        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)]
        for step, (yi,) in zip(range(steps), dataloader((_ys,), batch_size, key=loader_key)):
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, opt_state)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
    


if __name__ == "__main__":

    root = get_root_dir()

    # get config
    parser = HfArgumentParser(TraceHyps)
    config = parser.parse_args_into_dataclasses()[0]
    if config.trace_layer is None: raise ValueError("Pass in layer to get trace reps for")
    # make sure step_strategy compatible with length_strategy
    assert len(config.step_strategy) == len(config.length_strategy)
    # make sure step_strategy adds to 1
    assert sum(config.step_strategy) == 1

    # keys
    key = jr.PRNGKey(config.seed)
    model_key, loader_key = jr.split(key, 2)

    # download trace data if not already downloaded
    #trace_dir = download_layer_traces(root, config)

    # process data if not already processed
    segment_representations(root, config)

    # get segment representations
    # list of tensors with different lengths
    seg_reps = load_segment_representations(root, config)
    data_size = seg_reps[0].shape[1]

    # TODO : lower data dimensionality

    # node and optimizer
    model = NeuralODE(config, data_size, key=model_key)
    optim = optax.adabelief(config.lr)

    # train
    train(config, model, optim)






    

    

    