from dataclasses import dataclass, field
import random

from torch.utils.data import DataLoader

from datasets import load_dataset, DatasetDict
from transformers import HfArgumentParser


def prune_wait(x):
    trace = x['trace']
    seqs = trace.split('\n\n')
    new_seqs = []
    for seq in seqs:
        if 'wait' not in seq and 'hmm' not in seq:
            new_seqs.append(seq)
    new_trace = '\n\n'.join(new_seqs)
    x['trace'] = new_trace
    return x


@dataclass
class DataArguments:

    dataset_name: str = field(default=None)
    dataset_split: str = field(default="train")
    seed: int = field(default=42)


if __name__ == "__main__":

    data_args = HfArgumentParser((DataArguments)).parse_args_into_dataclasses()[0]

    # set seed
    random.seed(data_args.seed)

    # load dataset
    dataset = load_dataset(data_args.dataset_name)

    # process datasets
    train_dataset = dataset["train"].map(prune_wait, batched=False)
    eval_dataset = dataset["test"].map(prune_wait, batched=False)

    # pack into dataset
    dataset = DatasetDict({
        "train": train_dataset,
        "test": eval_dataset,
    })

    # push to hub
    dataset_name = data_args.dataset_name.split("/")[-1] + "_no_wait"
    dataset.push_to_hub("Ujan/"+dataset_name)

