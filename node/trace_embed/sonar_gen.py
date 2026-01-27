import sys
sys.path.append('../../')
from dataclasses import dataclass, field
from utils import get_root_dir

import re
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
    min_sent_chars: str = field(default=15)
    min_num_sents: int = field(default=10)


def merge_small_sentences(config, sentences):
    valid_sentences = []
    for sentence in sentences:
        # push first sentence
        if len(valid_sentences) == 0:
            valid_sentences.append(sentence)
        else:
            if len(valid_sentences[-1]) < config.min_sent_chars:
                # we have already removed trailing whitespaces
                valid_sentences[-1] += ' ' + sentence
            else:
                valid_sentences.append(sentence)
    return valid_sentences


# splits trace into sentences
def get_sentences(config, trace):
    # (?<![.?!:]) -> Negative lookbehind: Don't match if preceded by another punctuation mark
    # [.?!:]      -> Character class: Match any one of these four characters
    # (?=\s|$)    -> Positive lookahead: Match only if followed by space or end of string
    pattern = r'(?<![.?!:])[.?!:](?=\s|$)'
    sentences = re.split(pattern, trace)
    # clean up whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    # merge small sentences
    sentences = merge_small_sentences(config, sentences)
    return sentences


# TODO : reconstruct sentences
# if reconstruction poor, split large senteces until reconstruction acceptable
def validate_sentences(config, sentences, t2vec_model, vec2text_model):
    pass



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

    # sonar models
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

    # example
    trace = trace_dataset[0]['trace']
    sentences = get_sentences(config, trace)
    embeddings = t2vec_model.predict(["n²(1 + log n) ≤ C [n³ log n - (n - 1)^3 log n]"], source_lang="eng_Latn")
    print(embeddings.shape)
    print('\n\n')
    reconstructed = vec2text_model.predict(embeddings, target_lang="eng_Latn", max_seq_len=256)
    print(reconstructed)
    quit()






    

    

    