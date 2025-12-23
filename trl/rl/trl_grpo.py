import sys
sys.path.append('../../')
from utils import get_root_dir, process_rl_dataset
from rewards import get_reward_functions

from rl_config import TRLRLHyps
import logging
import warnings

from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    logging as transformers_logging
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from trl import (
    GRPOConfig,
    GRPOTrainer,
)


if __name__ == "__main__":

    root = get_root_dir()

    # get config
    parser = HfArgumentParser(TRLRLHyps)
    config = parser.parse_args_into_dataclasses()[0]

    # check values
    if config.model_name is None:
        raise ValueError("Model name must be specified")
    if config.use_vllm:
        if config.vllm_mode == 'server':
            if config.vllm_port is None:
                raise ValueError("VLLM port must be specified")
            print('Make sure vllm server is running at port : {}'.format(config.vllm_port))
        if config.vllm_mode == 'colocate':
            if config.vllm_max_model_length is None:
                config.vllm_max_model_length = config.max_prompt_len + config.max_gen_len
                print('gpu util set to {}'.format(config.vllm_gpu_memory_utilization))
                print('max model length set to {}'.format(config.vllm_max_model_length))

    # get model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision = config.model_revision,
        dtype = "auto",
        #device_map="auto",
        #attn_implementation='flash_attention_2',
    )
    if config.tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # PEFT
    if config.use_peft:
        print("Using PEFT")
        lora_config = LoraConfig(
            task_type = "CAUSAL_LM",
            r = config.lora_rank,
            lora_alpha = config.lora_alpha,
            # target_modules = all -> lora without regret
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        #model.print_trainable_parameters()

    # dataset
    dataset = load_dataset(config.dataset, split=config.dataset_split)
    # process dataset
    rl_dataset = dataset.map(process_rl_dataset, load_from_cache_file=False).remove_columns(['problem'])  
    rl_dataset = rl_dataset.rename_column('answer', 'solution')
    #print(tokenizer.apply_chat_template(rl_dataset[0]['prompt'], tokenize=False, add_generation_prompt=True))

    # set output dir
    model_name = config.model_name.split("/")[-1]
    checkpoint_folder = model_name + '_polaris'
    output_dir = root+"/"+config.output_dir+"/"+checkpoint_folder

    # TODO : sampling params -> default max_tokens : 2048 

    # reward functions
    reward_functions = get_reward_functions(config)
    if len(reward_functions) < 1: 
        raise ValueError("Need at least one reward function")

    # grpo config
    training_args = GRPOConfig(

        # context parallelism
        # For cp_size=2: use pad_to_multiple_of=4 (since cp_size * 2 = 4)
        # For cp_size=4: use pad_to_multiple_of=8 (since cp_size * 2 = 8)
        #pad_to_multiple_of = config.pad_to_multiple_of,

        # training args
        max_completion_length = config.max_gen_len,
        num_generations = config.num_gens,
        # `max_prompt_length` argument is deprecated 
        per_device_train_batch_size = config.per_device_train_batch_size,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        num_train_epochs = config.num_train_epochs,
        gradient_checkpointing = False,
        ddp_find_unused_parameters = False,

        # optim args
        learning_rate = config.learning_rate,
        warmup_steps = config.warmup_steps,
        lr_scheduler_type = config.lr_scheduler_type,

        # logging and saving
        logging_steps = config.logging_steps,
        save_strategy = "steps",
        save_steps = config.save_steps,
        save_total_limit = config.save_total_limit,
        optim = config.optim,
        report_to = "tensorboard", 
        output_dir = output_dir,

        # vLLM
        use_vllm = config.use_vllm,
        vllm_mode = config.vllm_mode,
        # general
        vllm_server_timeout = config.vllm_timeout,
        vllm_importance_sampling_correction = config.vllm_importance_sampling_correction,
        vllm_importance_sampling_cap = config.vllm_importance_sampling_cap,
        vllm_model_impl = config.vllm_model_impl,
        # server mode
        vllm_server_host = config.vllm_host,
        vllm_server_port = config.vllm_port,
        # colocate mode
        vllm_gpu_memory_utilization = config.vllm_gpu_memory_utilization,
        vllm_max_model_length = config.vllm_max_model_length,
        vllm_tensor_parallel_size = config.vllm_tensor_parallel_size,
        vllm_enable_sleep_mode = config.vllm_enable_sleep_mode,
    )

    # training
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=rl_dataset,
    )

    # supress warnings
    logging.basicConfig(level=logging.WARNING) # Set global logging level to WARNING
    logging.getLogger("vllm").setLevel(logging.WARNING)  # Silence INFO logs from vLLM
    transformers_logging.set_verbosity_warning() # Set Transformers logging to WARNING
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

        

