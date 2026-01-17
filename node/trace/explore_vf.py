import sys
sys.path.append('../../')
from dataclasses import dataclass, field

from utils import get_root_dir

from transformers import HfArgumentParser
from process_trace import load_segment_representations

import jax.numpy as jnp
from node import load_node

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class TraceHyps:

    seed: int = 42
    model_name: str = field(default="Qwen/Qwen3-4B-Thinking-2507")
    dataset_name: str = field(default="HuggingFaceH4/MATH-500")
    trace_layer: int = field(default=None)
    trace_dir: str = field(default="/data/Trace-Tensors")


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
                

if __name__ == "__main__":

    root = get_root_dir()

    # get config
    parser = HfArgumentParser(TraceHyps)
    config = parser.parse_args_into_dataclasses()[0]
    if config.trace_layer is None: raise ValueError("Pass in layer to get trace reps for")
    print('Warning :  Make sure layer is same as the layer data model was trained on')

    # get segment representations
    # list of tensors with different lengths
    data, metadatas = load_segment_representations(root, config)
    print('Num examples : {}'.format(len(data)))
    print('Avg num segments : {}'.format(int(sum([d.shape[0] for d in data])/len(data))))
    
    # node
    node = load_node(
        root + '/models/' + config.dataset_name.split("/")[-1] 
        + "-" + config.model_name.split("/")[-1]
        + "-layer-" + str(config.trace_layer) + ".eqx"
    )
    
    ####
    x_id = 76

    divs = jnp.array([node.vf.div(None, seg_rep, None) for seg_rep in data[x_id]])
    indices = jnp.argsort(divs).tolist()
    print(metadatas[x_id]['segment_texts'].split("<SEP>")[indices[0]])
    print('\n\n')
    print(metadatas[x_id]['segment_texts'].split("<SEP>")[indices[-1]])










    

    

    