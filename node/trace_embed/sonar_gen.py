import sys
sys.path.append('../../')
from dataclasses import dataclass, field
from utils import get_root_dir, combine_deepmath

import re
import numpy as np
import torch
import torch.nn.functional as F

import time
from multiprocessing import shared_memory
from multiprocessing.connection import Listener

from transformers import HfArgumentParser
from datasets import load_dataset

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
    og_embed = t2vec_model.predict(sentences, source_lang="eng_Latn")
    recon_sents = vec2text_model.predict(og_embed, target_lang="eng_Latn", max_seq_len=512)
    recon_embed = t2vec_model.predict(recon_sents, source_lang="eng_Latn")
    cos_sim = F.cosine_similarity(og_embed, recon_embed)
    print(len(sentences))
    print(len(recon_sents))
    print(cos_sim.shape)
    print(cos_sim)
    quit()



if __name__ == "__main__":

    root = get_root_dir()

    # get config
    parser = HfArgumentParser(SonarHyps)
    config = parser.parse_args_into_dataclasses()[0]

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

    # download deepmath trace data
    trace_dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    
    # merge deepmath traces
    trace_dataset = trace_dataset.map(
        combine_deepmath,
        batched=True,
        remove_columns=trace_dataset.column_names,
    ).shuffle(config.seed)

    # producer loop
    for example in trace_dataset:
        # get sentences from traces
        sentences = get_sentences(config, example['trace'])
        # quality check
        sentences = validate_sentences(config, sentences, t2vec_model, vec2text_model)






    

    

    