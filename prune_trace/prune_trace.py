import sys
sys.path.append("..")
from dataclasses import dataclass, field
import random
from utils import (
    process_sft_dataset
)

from lt_signals import LTSignals

from torch.utils.data import DataLoader

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)


@dataclass
class DataArguments:

    dataset_name: str = field(default=None)
    add_think: bool = field(default=False)
    seed: int = field(default=42)


@dataclass
class ModelArguments:

    model_name: str = field(default="Qwen/Qwen3-4B-Thinking-2507")
    batch_size: int = field(default=4)
    segment_length: int = field(default=128)
    overlap_segments: bool = field(default=False)
    preserve_last_segment: bool = field(default=False) # first segment always preserved
    stride: int = field(default=32)
    layers: str = field(default="all")
    sparsity: float = field(default=0.5)
    pruning_logic: str = field(default='algn')
    

def main(data_args, model_args):

    # set seed
    random.seed(data_args.seed)

    # load the tokenizer and the model
    model_args.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    model_args.model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        dtype="auto",
        device_map="auto"
    )

    # load dataset
    dataset = load_dataset(data_args.dataset_name)

    # process datasets
    train_dataset = process_sft_dataset(dataset["train"], model_args.tokenizer, data_args)
    eval_dataset = process_sft_dataset(dataset["test"], model_args.tokenizer, data_args)

    # dataloader    
    train_dataloader = DataLoader(train_dataset, batch_size=model_args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=model_args.batch_size)

    # lt object
    lts = LTSignals(model_args)
    
    # prune dataset
    train_dataset = lts.prune_dataset(train_dataloader)
    eval_dataset = lts.prune_dataset(eval_dataloader)

    # pack into dataset
    dataset = DatasetDict({
        "train": train_dataset,
        "test": eval_dataset,
    })

    return dataset


if __name__ == "__main__":

    data_args, model_args = HfArgumentParser((DataArguments, ModelArguments)).parse_args_into_dataclasses()

    # check args
    if data_args.dataset_name is None:
        raise ValueError("Spceify dataset")
    if model_args.overlap_segments:
        raise NotImplementedError(
            "Not supported curently for pruning. Need to map to non overlapping segments before pruning"
        )
    
    # generate pruned dataset
    pruned_dataset = main(data_args, model_args)

    # save pruned dataset
    model_name = model_args.model_name.split('/')[-1]
    dataset_name = 'lts_pruned_processed_'+data_args.dataset_name.split('/')[-1]+'_'+model_name+'_sparsity_'+str(model_args.sparsity)
    pruned_dataset.push_to_hub('Ujan/'+dataset_name)
    print('Done.')
