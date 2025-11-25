from dataclasses import dataclass, field
import random
import sys
sys.path.append('../')
from trl_utils import SYSTEM_PROMPT

from math_verify import parse, verify
from tqdm.auto import tqdm

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import HfArgumentParser
from transformers.utils import logging
logging.set_verbosity_error()


def create_prompt(question):
    messages =  [
        {"role" : "system",    "content" : SYSTEM_PROMPT},
        {"role" : "user", "content" : question},
    ]
    return messages


def get_accuracy(gold, answer):
    if verify(parse(gold), parse(answer)):
        return 1.0
    else:
        return 0.0


@dataclass
class DataArguments:

    sample: bool = field(default=False)
    num_samples: int = field(default=100)
    seed: int = field(default=42)


@dataclass
class ModelArguments:

    model_name: str = field(default=None)


if __name__ == "__main__":

    # parse cl arguments
    parser = HfArgumentParser((DataArguments, ModelArguments))
    data_args, model_args = parser.parse_args_into_dataclasses()

    # set seed
    random.seed(data_args.seed)

    # load dataset
    dataset_name = 'HuggingFaceH4/MATH-500'
    dataset = load_dataset(dataset_name)['test']

    # sample dataset
    if data_args.sample:
        dataset = dataset.select(
            random.sample(list(range(len(dataset))), data_args.num_samples)
        )

    # vllm object
    llm = LLM(model=model_args.model_name)

    # eval
    bar = tqdm(range(len(dataset)))
    accuracy = 0.0
    for example in dataset:
        
        question = example['problem']
        answer = example['solution']

        # prompt
        prompt = create_prompt(question)

        # query model
        outputs = llm.chat(
            prompt,
            #sampling_params=sampling_params,
            #chat_template=chat_template,
            use_tqdm=False
        )
        solution = outputs[0].outputs[0].text

        # get accuracy
        acc = get_accuracy(answer, solution)
        accuracy += acc

        bar.update(1)

    print(accuracy/len(dataset))
