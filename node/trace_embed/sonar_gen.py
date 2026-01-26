import sys
sys.path.append('../../')
from dataclasses import dataclass, field
from utils import get_root_dir

import time
import torch
import numpy as np
from multiprocessing import shared_memory
from multiprocessing.connection import Listener

from transformers import HfArgumentParser
from datasets import load_dataset
from utils import combine_deepmath

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)


@dataclass
class SonarHyps:

    # seed
    seed: int = 42

    # trace data
    dataset_name: str = field(default="zwhe99/DeepMath-103K")
    dataset_split: str = field(default="train")
    sample: bool = field(default=False)
    num_samples: int = field(default=500)
    # TODO : sentence instead since sonar does not work with long text?
    segment_by: str = field(default="\n\n")
    min_segment_chars: str = field(default=50)
    min_num_segments: int = field(default=10)


if __name__ == "__main__":

    root = get_root_dir()

    # get config
    parser = HfArgumentParser(SonarHyps)
    config = parser.parse_args_into_dataclasses()[0]

    # download deepmath trace data
    trace_dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    
    # format deepmath
    trace_dataset = trace_dataset.map(
        combine_deepmath,
        batched=True,
        remove_columns=trace_dataset.column_names,
    ).shuffle(config.seed)

    # TODO : sonar -> sentence as segment?
    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    vec2text_model = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    segs = trace_dataset[0]['trace'].split(config.segment_by)
    #print(segs[1])
    #print('\n\n')
    embeddings = t2vec_model.predict(["T(n) defined by T(n) = T(n-1) + n² + n² log n. Hmm, let's start by understanding what this function does."], source_lang="eng_Latn")
    print(embeddings.shape)
    print('\n\n')
    reconstructed = vec2text_model.predict(embeddings, target_lang="eng_Latn", max_seq_len=256)
    print(reconstructed)
    quit()


    segment_representations(root, config)

    # get segment representations
    # list of tensors with different lengths
    data, metadatas = load_segment_representations(root, config)
    print('Num examples : {}'.format(len(data)))
    print('Avg num segments : {}'.format(int(sum([d.shape[0] for d in data])/len(data))))

    # lower data dimensionality before training
    if config.proj_before_train:
        print('PCA before training...')
        pca, data = pca_trace(data, num_components=config.proj_dim)
    
    # node and optimizer
    model = NeuralODE(
        config.node_width, config.node_depth, data[0].shape[1], 
        config.pid_rtol, config.pid_atol, key=model_key
    )
    optim = optax.adabelief(config.lr)

    # train
    # either integrate per trajectory -> this is being done now!
    # or common time stamps, mask invalid time stamps
    # TODO : or latent ode
    train(root, config, model, data, optim, loader_key)






    

    

    