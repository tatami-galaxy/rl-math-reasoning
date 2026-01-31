import sys
sys.path.append('../')
sys.path.append('../../')
from dataclasses import dataclass, field

import time
from multiprocessing import shared_memory
from multiprocessing.connection import Client

from utils import get_root_dir

from transformers import HfArgumentParser

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx 
from node import NeuralODE, save_node
import optax

ADDRESS = ("localhost", 9000)
AUTHKEY = b"secret"


@dataclass
class TraceHyps:

    # seed
    seed: int = 42

    # neural ode
    trace_embed: int = field(default=1024)
    node_depth: int = field(default=3)
    node_width: int = field(default=64)
    pid_rtol: float = field(default=1e-3)
    pid_atol: float = field(default=1e-6)

    # training
    lr: float = field(default=1e-3) # 2e-5
    total_steps: int = field(default=1000)
    step_strategy: list[float] = field(default_factory=lambda: [0.5, 0.5])
    length_strategy: list[float] = field(default_factory=lambda: [0.1, 1])
    min_trace_length: int = field(default=50) # in num sentences
    log_steps: int = field(default=50)
    plot: bool = field(default=False)
    plot_steps: int = field(default=100)
    save_after_train: bool = field(default=False)


@eqx.filter_value_and_grad
def grad_loss(model, ts, ys):
    #y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
    # integrate with node from the first time step
    # then calculate loss at each time step
    y_pred = model(ts, ys[0])
    return jnp.mean((ys - y_pred) ** 2)


@eqx.filter_jit
def make_step(ts, ys, model, optim, opt_state):
    loss, grads = grad_loss(model, ts, ys)
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


if __name__ == "__main__":

    root = get_root_dir()

    # get config
    parser = HfArgumentParser(TraceHyps)
    config = parser.parse_args_into_dataclasses()[0]
    # make sure step_strategy compatible with length_strategy
    assert len(config.step_strategy) == len(config.length_strategy)
    # make sure step_strategy adds to 1
    assert sum(config.step_strategy) == 1

    # keys
    model_key = jr.PRNGKey(config.seed)

    # node and optimizer
    model = NeuralODE(
        config.node_width, config.node_depth, config.trace_embed, 
        config.pid_rtol, config.pid_atol, key=model_key
    )
    optim = optax.adabelief(config.lr)

    # connect to producer
    conn = Client(ADDRESS, authkey=AUTHKEY)
    print("Trainer: connected to Sonar")

    # outer train loop
    # initially we train on only the first n% of each time series.
    # a standard trick to avoid getting caught in a local minimum.
    for step_frac, length_frac in zip(config.step_strategy, config.length_strategy):

        # num steps according to step strategy
        steps = int(config.total_steps * step_frac)

        # optimizer state
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        # inner train loop
        step = 0
        while step < steps:

            # receive sonar embeddings
            msg = conn.recv()
            shm = shared_memory.SharedMemory(name=msg["shm_name"])
            batch_embeddings = []
            for meta in msg["embeddings"]:
                embed = np.ndarray(
                    shape=tuple(meta["shape"]),
                    dtype=np.dtype(meta["dtype"]),
                    buffer=shm.buf,
                    offset=meta["offset"],
                )
                batch_embeddings.append(jnp.asarray(embed)) 
                step += 1
            # relieve producer to produce next batch
            shm.close()
            conn.send("done") # -> TODO :; debug this

            print('HERE')

            # train steps for received data
            # TODO :
            for embedding in batch_embeddings:
                ts = jnp.arange(embedding.shape[0])
                ys = embedding
                # length strategy
                # if total length already small do not truncate
                if int(length_frac * ys.shape[0]) >= config.min_trace_length:
                    ts = jnp.arange(int(length_frac * ys.shape[0]))
                    ys = ys[:int(length_frac * ys.shape[0])]
                else:
                    ts = jnp.arange(ys.shape[0])
                # train step
                start = time.time()
                loss, model, opt_state = make_step(ts, ys, model, optim, opt_state)
                print("loss: {}".format(loss))
                quit()
                end = time.time()

                # log and plot
                if (step % config.log_steps) == 0 or step == steps - 1:
                    print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")

    # save 
    if config.save_after_train:
        print("Saving model...")
        hyperparams = {
            "seed": config.seed,
            "node_width": config.node_width,
            "node_depth": config.node_depth,
            "data_size": data[0].shape[1],
            "pid_rtol": config.pid_rtol,
            "pid_atol": config.pid_atol,
        }
        filename = root + "/models/"
        filename += config.dataset_name.split("/")[-1] + "-" + config.model_name.split("/")[-1]
        filename += "-layer-" + str(config.trace_layer) + ".eqx"
        save_node(filename, hyperparams, model)





    

    

    