import sys
sys.path.append('../../')
from dataclasses import dataclass, field
import time

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
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


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
    min_segments: int = field(default=5)
    resegment: bool = field(default=False)

    # neural ode
    node_depth: int = field(default=3)
    node_width: int = field(default=64)
    pid_rtol: float = field(default=1e-3)
    pid_atol: float = field(default=1e-6)
    total_steps: int = field(default=1000)
    step_strategy: list[float] = field(default_factory=lambda: [0.5, 0.5])
    length_strategy: list[float] = field(default_factory=lambda: [0.1, 1])
    min_trace_length: int = field(default=10) # in num segments

    # training
    lr: float = field(default=1e-3)
    log_steps: int = field(default=50)


@eqx.filter_value_and_grad
def grad_loss(model, ti, yi):
    #y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
    # integrate with node from the first time step
    # then calculate loss at each time step
    y_pred = model(ti, yi[0])
    return jnp.mean((yi - y_pred) ** 2)


@eqx.filter_jit
def make_step(ti, yi, model, opt_state):
    loss, grads = grad_loss(model, ti, yi)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def data_sampler(data, *, key):
    dataset_size = len(data)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        index = 0
        while index < dataset_size:
            ex_index = perm[index]
            yield data[ex_index]
            index += 1


def train(config, model, data, optim):
    # Up until step 500 we train on only the first 10% of each time series.
    # This is a standard trick to avoid getting caught in a local minimum.
    for step_frac, length_frac in zip(config.step_strategy, config.length_strategy):

        # num steps according to step strategy
        steps = int(config.total_steps * step_frac)

        # optimizer state
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        # train loop
        for step, yi in zip(range(steps), data_sampler(data, key=loader_key)):
            # length strategy
            # if total length already small do not truncate
            if int(length_frac * yi.shape[0]) >= config.min_trace_length:
                ts = jnp.arange(int(length_frac * yi.shape[0]))
                yi = yi[:int(length_frac * yi.shape[0])]
            else:
                ts = jnp.arange(yi.shape[0])
            # train step
            start = time.time()
            loss, model, opt_state = make_step(ts, yi, model, opt_state)
            end = time.time()

            if (step % config.log_steps) == 0 or step == steps - 1:
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
    data, metadata = load_segment_representations(root, config)
    print('Num examples : {}'.format(len(data)))
    print('Avg num segments : {}'.format(int(sum([d.shape[0] for d in data])/len(data))))

    ###
    # TODO : lower data dimensionality
    data_all = np.concatenate([np.array(d) for d in data], axis=0)
    pca = PCA(n_components=2)
    pca.fit(data_all)
    compressed_series = [pca.transform(d) for d in data][:3]

    # plot
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    for i, ts in enumerate(compressed_series):
        #t = np.arange(len(ts))
        t = np.linspace(0, 1, len(ts))
        color = colors[i % len(colors)]

        ax.plot(
            ts[:, 0],      # PCA component 1
            ts[:, 1],      # PCA component 2
            t,             # time axis
            color=color,
            alpha=0.8
        )

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_zlabel("Time step")
    ax.set_title("2D PCA trajectories over time (variable length)")

    plt.tight_layout()
    plt.savefig("pca_2d_trajectories_3d.jpg", dpi=300, bbox_inches="tight")
    plt.show()

    quit()
    ###

    # node and optimizer
    model = NeuralODE(config, data[0].shape[1], key=model_key)
    optim = optax.adabelief(config.lr)

    # train
    # either integrate per trajectory
    # or common time stamps, mask invalid time stamps
    train(config, model, data, optim)






    

    

    