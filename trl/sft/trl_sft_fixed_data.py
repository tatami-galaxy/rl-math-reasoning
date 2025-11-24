# TODO : same data size

import os
import sys
sys.path.append("..")
from trl_sft_config import TRLSFTHyps
from trl_utils import SYSTEM_PROMPT, REASONING_START
from trl_utils import get_root_dir, create_chat_template
from trl_sft_config import TRLSFTHyps

from functools import partial

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


class NewSFTTrainer(SFTTrainer):  
    def _save_checkpoint(self, model, trial):  
        super()._save_checkpoint(model, trial)  # Default saving  
        trainer_state_path = os.path.join(self.args.output_dir, 'trainer_state.json')  
        self.state.save_to_json(trainer_state_path)  



# format dataset for sft
def format_dataset(x, tokenizer, add_think=False):

    def process_trace(q, trace, add_think):
        think_token = REASONING_START if add_think else ''
        messages =  [
            {"role" : "system",    "content" : SYSTEM_PROMPT},
            # add <think> token after question
            {"role" : "user",      "content" : q+think_token},
            {"role" : "assistant", "content" : trace},
        ]
        # no generation_prompt because this is sft, needed for rl
        return tokenizer.apply_chat_template(messages, tokenize=False) 

    new_examples = {
        "text": [],
        "answer": [],
    }
    # all 3 traces
    for i, question in enumerate(x["question"]):
        answer = x["final_answer"][i]
        traces = [x["r1_solution_1"][i], x["r1_solution_2"][i], x["r1_solution_3"][i]]
        for trace in traces:
            new_examples["text"].append(process_trace(question, trace, add_think))
            new_examples["answer"].append(answer)

    return new_examples


def main():

    root = get_root_dir()

    # get hyps
    parser = HfArgumentParser(TRLSFTHyps)
    config = parser.parse_args_into_dataclasses()[0]
    if config.model_name is None:
        raise ValueError("model name must be specified.")
    if config.max_seq_len is None:
        raise ValueError("max sequence length must be specified.")

    print("cp size set to {}. Modify accelerate config and sft config to change".format(config.pad_to_multiple_of//2))

    # Load model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        dtype="auto",
        #device_map="auto",
        #attn_implementation='flash_attention_2',
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # create or modify chat template
    if tokenizer.chat_template is None:
        print("No chat template found. Creating custom chat template...")
        tokenizer = create_chat_template(tokenizer)
        config.add_think = True
    else:
        print("Chat template found.")

    # load dataset
    dataset = load_dataset(config.sft_dataset, split=config.sft_dataset_split)

    # format dataset to chat template
    # create 3 examples from each deepmath example
    # with the 3 given reasoning traces
    dataset = dataset.map(
        partial(
            format_dataset,
            tokenizer=tokenizer,
            add_think=config.add_think,
        ),
        batched = True,
        remove_columns=dataset.column_names,
    ).shuffle(config.seed)

    # truncate fine-tuning dataset to max_seq_len
    # seq_len: 16384 -> dataset: 200387 
    # seq_len: 8192 -> dataset: 62255
    # seq_len: 4096 -> dataset: 1877
    # text already has chat template applied
    print(dataset)
    dataset = dataset.filter(
        lambda x: len(tokenizer(x["text"])['input_ids']) <= config.max_seq_len
    )
    print(dataset)
    quit()

    # split dataset into train and eval
    dataset = dataset.train_test_split(test_size=config.eval_dataset_size, seed=config.seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # sample for testing
    if config.sample and len(train_dataset) > config.num_samples:
        train_dataset = train_dataset.select(range(config.num_samples))
        eval_dataset = eval_dataset.select(range(int(config.num_samples * config.eval_dataset_size)))

    # set output directory
    model_name = config.model_name.split("/")[-1]
    dataset_name = config.sft_dataset.split("/")[-1]
    seq_len = config.max_seq_len
    checkpoint_folder = model_name + "_" + dataset_name + "_seq_" + str(seq_len)
    output_dir = root+"/"+config.output_dir+"/"+checkpoint_folder
    
    # Train
    trainer = NewSFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        args = SFTConfig(
            # dataset
            dataset_text_field = "text",

            # context parallelism
            # For cp_size=2: use pad_to_multiple_of=4 (since cp_size * 2 = 4)
            # For cp_size=4: use pad_to_multiple_of=8 (since cp_size * 2 = 8)
            pad_to_multiple_of = config.pad_to_multiple_of, # ensures divisibility by cp_size * 2
            max_length = config.max_seq_len,
            #packing=True,   # use packing to reduce padding -> needs flash attention
            #use_liger_kernel=True,  # compatible with CP
            per_device_train_batch_size = config.per_device_train_batch_size,
            per_device_eval_batch_size = config.per_device_eval_batch_size,
            # The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg can't be set to True simultaneously
            gradient_checkpointing = False,
            gradient_accumulation_steps = config.gradient_accumulation_steps, # Use GA to mimic batch size

            # training args
            warmup_steps = config.warmup_steps,
            num_train_epochs = config.num_train_epochs, # do 1 epoch
            learning_rate = config.learning_rate, # 2e-5 with constant schedule
            logging_steps = config.logging_steps,
            eval_strategy = "steps",
            eval_steps=config.eval_steps,
            save_steps = config.save_steps,
            save_total_limit = config.save_total_limit,
            optim = config.optim,
            weight_decay = config.weight_decay,
            lr_scheduler_type = config.lr_scheduler_type,
            seed = config.seed,
            report_to = "tensorboard", 
            output_dir=output_dir
        ),
    )

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)



if __name__ == "__main__":
    main()
