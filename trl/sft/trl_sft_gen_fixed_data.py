import sys
sys.path.append("..")
from trl_sft_config import TRLSFTHyps

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)


def format_dataset(x):

    new_examples = {
        "answer": [],
        "question": [],
        "trace": [],
        "difficulty": [],
        "topic": [],
    }
    # all 3 traces
    for i, question in enumerate(x["question"]):
        traces = [x["r1_solution_1"][i], x["r1_solution_2"][i], x["r1_solution_3"][i]]
        for trace in traces:
            new_examples["answer"].append(x["final_answer"][i])
            new_examples["question"].append(question)
            new_examples["trace"].append(trace)
            new_examples["difficulty"].append(x["difficulty"][i])
            new_examples["topic"].append(x["topic"][i])

    return new_examples


def main():

    # get hyps
    parser = HfArgumentParser(TRLSFTHyps)
    config = parser.parse_args_into_dataclasses()[0]
    if config.model_name is None:
        raise ValueError("model name must be specified for tokenizer.")
    if config.max_seq_len is None:
        raise ValueError("max sequence length must be specified.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # load dataset
    dataset = load_dataset(config.sft_dataset, split=config.sft_dataset_split)

    # create 3 examples from each deepmath example
    # with the 3 given reasoning traces
    dataset = dataset.map(
        format_dataset,
        batched=True,
        remove_columns=dataset.column_names,
    ).shuffle(config.seed)

    # filter by trace length, not chat template
    dataset = dataset.filter(
        lambda x: len(tokenizer(x["trace"])['input_ids']) <= config.max_seq_len
    )

    # split dataset into train and eval
    dataset = dataset.train_test_split(test_size=config.total_eval_samples, seed=config.seed)
    train_dataset = dataset["train"]
    train_dataset = train_dataset.select(range(config.total_train_samples))
    eval_dataset = dataset["test"]

    # upload dataset
    dataset = DatasetDict({
        "train": train_dataset,
        "test": eval_dataset,
    })
    dataset_name = "_DeepMath-103K_samples_"+str(config.total_train_samples)+ "_seq_"+str(config.max_seq_len)
    dataset.push_to_hub("Ujan/"+dataset_name)


if __name__ == "__main__":
    main()
