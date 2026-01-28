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


def pca_trace(data, num_components=2):
    data_all = np.concatenate([np.array(d) for d in data], axis=0)
    pca = PCA(n_components=num_components)
    pca.fit(data_all)
    compressed_data = [pca.transform(d) for d in data]
    return pca, compressed_data


def plot_3d(data, pca, compressed_data, model, step, num_plots=2):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(len(compressed_data[:num_plots])):
        ys = data[i]
        ys_comp = compressed_data[i]
        t = np.linspace(0, 1, len(ys_comp))
        # data
        ax.plot(ys_comp[:, 0], ys_comp[:, 1], t, color="dodgerblue", alpha=0.8, label="data")
        # model
        y_pred_comp = pca.transform(model(jnp.arange(ys.shape[0]), ys[0]))
        ax.plot(y_pred_comp[:, 0], y_pred_comp[:, 1], t, color="crimson", alpha=0.8, label="model")

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_zlabel("Time step")
    ax.set_title("2D PCA trajectories over time")
    plt.tight_layout()
    plt.savefig("pca_2d_trajectories_3d_"+str(step)+".jpg", dpi=300, bbox_inches="tight")


def train(root, config, model, data, optim, loader_key):
    # Up until step 500 we train on only the first 10% of each time series.
    # This is a standard trick to avoid getting caught in a local minimum.

    # pca for viz
    if config.plot:
        pca, compressed_data = pca_trace(data, num_components=2)
    
    # training
    for step_frac, length_frac in zip(config.step_strategy, config.length_strategy):

        # num steps according to step strategy
        steps = int(config.total_steps * step_frac)

        # optimizer state
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        # train loop
        for step, ys in zip(range(steps), data_sampler(data, key=loader_key)):

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
            end = time.time()

            # log and plot
            if (step % config.log_steps) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
            if config.plot and (step % config.plot_steps) == 0 or step == steps - 1:
                    plot_3d(data, pca, compressed_data, model, step, num_plots=2)

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
    key = jr.PRNGKey(config.seed)
    model_key, loader_key = jr.split(key, 2)

    # node and optimizer
    model = NeuralODE(
        config.node_width, config.node_depth, config.trace_embed, 
        config.pid_rtol, config.pid_atol, key=model_key
    )
    optim = optax.adabelief(config.lr)

    # connect to producer
    conn = Client(ADDRESS, authkey=AUTHKEY)
    print("Trainer: connected to Sonar")

    # consume sonar embeddings
    while True:
        msg = conn.recv()
        shm = shared_memory.SharedMemory(name=msg["shm_name"])

        batch_embeddings_np = []
        for meta in msg["embeddings"]:
            embed = np.ndarray(
                shape=tuple(meta["shape"]),
                dtype=np.dtype(meta["dtype"]),
                buffer=shm.buf,
                offset=meta["offset"],
            )

            batch_embeddings_np.append(jnp.asarray(embed)) 

        print(f"Process B: received sample with {len(arrays)} arrays")

        # Simulate processing
        time.sleep(2)

        # Cleanup
        shm.close()
        conn.send("done")



    segment_representations(root, config)

    # get segment representations
    # list of tensors with different lengths
    data, metadatas = load_segment_representations(root, config)
    print('Num examples : {}'.format(len(data)))
    print('Avg num segments : {}'.format(int(sum([d.shape[0] for d in data])/len(data))))

    # train
    # either integrate per trajectory -> this is being done now!
    # or common time stamps, mask invalid time stamps
    # TODO : or latent ode
    train(root, config, model, data, optim, loader_key)






    

    

    