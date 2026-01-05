from dataclasses import dataclass, field


@dataclass
class TRLRLHyps:

    # seed
    seed: int = 42

    # Dataset
    dataset: str = field(default="POLARIS-Project/Polaris-Dataset-53K") # zwhe99/DeepMath-103K -> filtered by max trace len
    dataset_split: str = field(default="train")
    sample: bool = field(default=False)
    num_samples: int = field(default=1000)

    # Model parameters
    model_name: str = field(default=None)
    tokenizer_name: str = field(default=None)
    model_revision: str = field(default="main")

    # PEFT parameters
    use_peft: bool = field(default=False)
    lora_rank: int = field(default=4)
    lora_alpha: int = field(default=8) # 2*rank speeds up training (unsloth suggestion)

    # training parameters
    # batch size must be a multiple of num_generations to fit all generations of a prompt in a single batch
    # if batch_size == num_genarations, entire batch is completions from single prompt
    # elif batch_size == n * num_generations, batch is num_genetations completetions from each of the n prompts
    # TODO : 2048 -> enforce model to be more concise?
    max_gen_len: int = field(default=2048)  # 2048, 4096, 8192, 16384
    num_gens: int = field(default=8)
    max_prompt_len: int = field(default=256) # 512, TODO : deprecated
    proof_think: bool = field(default=False)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    gradient_checkpointing: bool = field(default=False)
    num_train_epochs: int = field(default=1)

    # reward function
    acc_reward: bool = field(default=False)
    think_reward: bool = field(default=False)
    overlong_reward: bool = field(default=False)

    # logging
    per_device_eval_batch_size: int = field(default=4)
    logging_steps: int = field(default=50)
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=2)
    resume_from_checkpoint: bool = field(default=False)
    output_dir: str = field(default="/models/trl/rl_checkpoints")

    # Optimizer parameters
    optim: str = field(default="adamw_torch_fused") # sgd
    learning_rate: float = field(default=2e-5)  # 1e-4, 2e-5 
    warmup_steps: int = field(default=8)
    lr_scheduler_type: str = field(default="constant")   # linear, constant

    # vLLM
    use_vllm: bool = field(default=False)
    vllm_model_impl: str = field(default="vllm") # transformers
    vllm_mode: str = field(default='server') # server, colocate
    vllm_importance_sampling_correction: bool = field(default=True)
    vllm_importance_sampling_cap: float = field(default=2.0)
    # server mode
    vllm_host: str = field(default="127.0.0.1")
    vllm_port: int = field(default=None)
    vllm_timeout: float = field(default=240.0)
    # colocate mode
    vllm_gpu_memory_utilization: float = field(default=0.3)
    vllm_max_model_length: int = field(default=None) # set it to > maximum prompt length + max_completion_length
    vllm_tensor_parallel_size: int = field(default=1)
    vllm_enable_sleep_mode: bool = field(default=False) # low mem, high latency

    # LEAN
    lean_server_timeout: float = field(default=1200.0)
