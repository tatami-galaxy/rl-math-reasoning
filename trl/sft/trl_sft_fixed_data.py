import os
import sys
sys.path.append("..")
from trl_sft_config import TRLSFTHyps
from utils import process_sft_dataset
from utils import get_root_dir, create_chat_template
from trl_sft_config import TRLSFTHyps

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import SFTTrainer, SFTConfig


class NewSFTTrainer(SFTTrainer):  
    def _save_checkpoint(self, model, trial):  
        super()._save_checkpoint(model, trial)  # Default saving  
        trainer_state_path = os.path.join(self.args.output_dir, 'trainer_state.json')  
        self.state.save_to_json(trainer_state_path)  


def main():

    root = get_root_dir()

    # get hyps
    parser = HfArgumentParser(TRLSFTHyps)
    config = parser.parse_args_into_dataclasses()[0]
    if config.model_name is None:
        raise ValueError("model name must be specified.")
    if config.max_seq_len is None:
        raise ValueError("max sequence length must be specified.")
    if config.processed_dataset is None:
        raise ValueError("processed dataset must be specified.")
    if config.processed_dataset.split('_')[-1] != str(config.max_seq_len):
        raise ValueError("seq len mismatch")

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
    dataset = load_dataset(config.processed_dataset) 
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # process dataset
    train_dataset = process_sft_dataset(train_dataset, tokenizer, config)
    eval_dataset = process_sft_dataset(eval_dataset, tokenizer, config)

    # set output directory
    model_name = config.model_name.split("/")[-1]
    dataset_name = config.processed_dataset.split("/")[-1]
    checkpoint_folder = model_name + '_' + dataset_name + '_epoch_' + str(config.num_train_epochs)
    output_dir = root+"/"+config.output_dir+"/"+checkpoint_folder
    
    # train
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
