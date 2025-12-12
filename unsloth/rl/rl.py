import sys
sys.path.append('../../')
import os
from os.path import dirname
import chz
import numpy as np

from rl_config import RLHyps
from utils import SYSTEM_PROMPT, create_chat_template

from math_verify import parse, verify
from datasets import load_dataset

from unsloth import FastLanguageModel
from safetensors import safe_open
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer


# Disable standby mode to avoid expandable_segments conflict
# os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"


def get_root_dir():
    root = os.path.abspath('')
    while root.split('/')[-1] != 'rl-math-unsloth':
        root = dirname(root)
    return root

# format dataset for sft
def format_dataset(x, system_prompt, tokenizer):
    problem = x["problem"]
    solution = x["generated_solution"]
    expected_answer = x["expected_answer"]
    messages =  [
        {"role" : "system",    "content" : system_prompt},
        {"role" : "user",      "content" : problem},
        {"role" : "assistant", "content" : solution},
    ]
    return {
        "text" : tokenizer.apply_chat_template(messages, tokenize=False),
        "expected_answer" : expected_answer,
    }


def extract_hash_answer(text):
    if "####" in text:
        return text.split("####")[1].strip()
    return text


def format_reward(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # if math-verify gets something
        if parse(response) is not None: score += 1.0
        scores.append(score)
    return scores


def accuracy_reward(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [parse(response) for response in responses]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):

        if guess is None:
            scores.append(-2.0)
            continue
        if verify(parse(true_answer), guess):
            scores.append(5.0)
        else:
            scores.append(-1.0)
    return scores



def main(config: RLHyps):

    # get project root dir
    root = get_root_dir()

    # load model
    if config.lora_config is not None:
        # check if LoRA is trained
        with safe_open(config.lora_config + "/adapter_model.safetensors", framework = "pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                n_zeros = (tensor == 0).sum() / tensor.numel()
                assert n_zeros.item() != tensor.numel(), "LoRA is not trained"
        model, tokenizer = FastLanguageModel.from_pretrained( 
            model_name=config.lora_config,
            max_seq_length=config.max_seq_len,
            fast_inference=config.fast_inference,
        )
    else:
        assert config.model_name is not None, "Model name must be provided"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = config.model_name,
            max_seq_length = config.max_seq_len,
            load_in_16bit = config.load_in_16bit,
            fast_inference = config.fast_inference, 
            max_lora_rank = config.lora_rank,   # does this get overridden?
            gpu_memory_utilization = config.gpu_memory_utilization
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = config.lora_rank,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha = config.lora_rank*2 if config.lora_alpha is None else config.lora_alpha, # *2 speeds up training
            use_gradient_checkpointing = "unsloth", # Reduces memory usage
            random_state = config.seed,
        )

    # create or modify chat template
    tokenizer = create_chat_template(tokenizer)

    # load rl dataset
    dataset = load_dataset(config.rl_dataset, config.rl_dataset_config, split = "train")

    # process dataset
    dataset = dataset.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    })

    # remove long prompts
    tokenized = dataset.map(
        lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
        batched = True,
    )
    tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    max_prompt_length = maximum_length + 1 # + 1 just in case!

    # filter only samples smaller than 90% max length
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])

    # GRPO trainer
    # completion length
    max_completion_length = config.max_seq_len - max_prompt_length
    # vllm sampling params
    #vllm_sampling_params = SamplingParams(
        #min_p = config.min_p,
        #top_p = config.top_p,
        #top_k = config.top_k,
        #seed = config.seed,
        #stop = [tokenizer.eos_token],
        #include_stop_str_in_output = True,
    #)
    # training args
    training_args = GRPOConfig(
        seed=config.seed,
        #vllm_sampling_params = vllm_sampling_params,
        temperature = config.temperature,
        min_p = config.min_p,
        top_p = config.top_p,
        top_k = config.top_k,
        learning_rate = config.learning_rate,
        weight_decay = config.weight_decay,
        warmup_ratio = config.warmup_ratio,
        lr_scheduler_type = config.lr_scheduler_type,
        #optim = config.optim,
        fp16 = config.fp16,
        logging_steps = config.logging_steps,
        per_device_train_batch_size = config.per_device_train_batch_size,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        num_generations = config.num_generations, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = config.max_steps,
        save_steps = config.save_steps,
        report_to = "tensorboard", # Can use Weights & Biases
        output_dir = root+"/"+config.output_dir,
        # For optional training + evaluation
        #fp16_full_eval = True,
        #per_device_eval_batch_size = 8,
        #eval_accumulation_steps = 1,
        #eval_strategy = "steps",
        #eval_steps = 1,
    )

    # Trainer
    # For optional training + evaluation
    #new_dataset = dataset.train_test_split(test_size = 0.01)
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            format_reward,
            accuracy_reward,
        ],
        args = training_args,
        train_dataset = dataset,
        # For optional training + evaluation
        #train_dataset = new_dataset["train"],
        #eval_dataset = new_dataset["test"],
    )

    # Train!
    trainer.train()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
