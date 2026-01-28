import sys
sys.path.append('../../')
from dataclasses import dataclass, field
from utils import get_root_dir, combine_deepmath

import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time
from multiprocessing import shared_memory
from multiprocessing.connection import Listener

from transformers import HfArgumentParser
from datasets import load_dataset

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

ADDRESS = ("localhost", 9000)
AUTHKEY = b"secret"


@dataclass
class SonarHyps:

    # seed
    seed: int = 42

    # trace data
    dataset_name: str = field(default="zwhe99/DeepMath-103K")
    dataset_split: str = field(default="train")
    sample: bool = field(default=False)
    num_samples: int = field(default=500)
    batch_size: int = field(default=64)
    segment_by: str = field(default="\n")
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
def get_sentences(config, batch):
    # (?<![.?!:]) -> Negative lookbehind: Don't match if preceded by another punctuation mark
    # [.?!:]      -> Character class: Match any one of these four characters
    # (?=\s|$)    -> Positive lookahead: Match only if followed by space or end of string
    pattern = r'(?<![.?!:])[.?!:](?=\s|$)'

    traces = batch['trace']
    batch_sentences = []
    batch_lengths = []
    for trace in traces:
        sentences = re.split(pattern, trace)
        # clean up whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        # merge small sentences
        sentences = merge_small_sentences(config, sentences)
        if len(sentences) >= config.min_num_sents:
            batch_sentences.append(sentences)
            batch_lengths.append(len(sentences))

    return batch_sentences, batch_lengths


# TODO : use recon as an objective in node training
def validate_sentences(config, sentences, t2vec_model, vec2text_model):
    print('validate_sentences function is only for testing!')
    og_embed = t2vec_model.predict(sentences, source_lang="eng_Latn")
    recon_sents = vec2text_model.predict(og_embed, target_lang="eng_Latn", max_seq_len=512)
    recon_embed = t2vec_model.predict(recon_sents, source_lang="eng_Latn")
    cos_sim = F.cosine_similarity(og_embed, recon_embed)
    indices = torch.argsort(cos_sim).tolist()


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
    # dont need decoder atm
    """vec2text_model = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device("cuda"),
        dtype=torch.float16,
    )"""

    # download deepmath trace data
    trace_dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    
    # merge deepmath traces
    trace_dataset = trace_dataset.map(
        combine_deepmath,
        batched=True,
        remove_columns=trace_dataset.column_names,
    ).shuffle(config.seed)

    # dataloader
    dataloader = DataLoader(
        trace_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    listener = Listener(ADDRESS, authkey=AUTHKEY)
    print("Sonar: waiting for trainer")
    conn = listener.accept()
    print("Sonar: connected")

    # producer loop
    for batch in dataloader:
        # get sentences from traces
        # might return less than batch size since we discard very small traces
        batch_sentences, batch_lengths = get_sentences(config, batch)
        if len(batch_sentences) == 0: continue
        # quality check
        #validate_sentences(config, batch_sentences[0], t2vec_model, vec2text_model)

        # concat sentences for generating embeddings
        batch_sentences_flat = [sentence for sentences in batch_sentences for sentence in sentences]
        # generate sonar embeddings
        embeddings = t2vec_model.predict(batch_sentences_flat, source_lang="eng_Latn")
        # split back
        batch_embeddings = torch.split(embeddings, batch_lengths)

        # convert to numpy
        batch_embeddings_np = [embedding.cpu().numpy() for embedding in batch_embeddings]

        # producer logic
        total_bytes = sum(a.nbytes for a in batch_embeddings_np )
        shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        metadata = []
        offset = 0

        # write arrays contiguously
        for embedding in batch_embeddings_np:
            buf = np.ndarray(embedding.shape, dtype=embedding.dtype, buffer=shm.buf, offset=offset)
            buf[:] = embedding
            metadata.append({
                "shape": embedding.shape,
                "dtype": str(embedding.dtype),
                "offset": offset,
                "nbytes": embedding.nbytes,
            })
            offset += embedding.nbytes

        # Send control message
        conn.send({
            "shm_name": shm.name,
            "arrays": metadata,
        })

        print(f"Sonar: sent embedding with {len(metadata)} arrays")

        # producer keeps shm alive until consumer is done
        ack = conn.recv()
        if ack == "done":
            shm.close()
            shm.unlink()







    

    

    