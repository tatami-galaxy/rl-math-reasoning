from dataclasses import dataclass, field


@dataclass
class TRLSFTHyps:

    # seed
    seed: int = 42

    # Dataset
    sft_dataset: str = field(default="zwhe99/DeepMath-103K")
    sft_dataset_split: str = field(default="train")
    processed_dataset: str = field(default=None)
    add_think: bool = field(default=False)
    total_train_samples: int = field(default=10000)
    total_eval_samples: int = field(default=1000)
    dataset_save_dir: str = field(default=None)


    # Model parameters
    model_name: str = field(default=None)
    model_revision: str = field(default="main")

    # Training parameters
    max_seq_len: int = field(default=None)  # 4096, 8192, 16384
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=4)
    # For cp_size=2: use pad_to_multiple_of=4 (since cp_size * 2 = 4)
    # For cp_size=4: use pad_to_multiple_of=8 (since cp_size * 2 = 8)
    pad_to_multiple_of: int = field(default=4)
    num_train_epochs: int = field(default=1)

    # Checkpointing and evaluation
    per_device_eval_batch_size: int = field(default=4)
    logging_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=3)
    resume_from_checkpoint: bool = field(default=False)

    # Optimizer parameters
    optim: str = field(default="adamw_torch_fused")
    learning_rate: float = field(default=2e-5)     # 1e-4, 2e-5 
    warmup_steps: int = field(default=8)
    lr_scheduler_type: str = field(default="constant")   # linear, constant
    weight_decay: float = field(default=0.01)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_eps: float = field(default=1e-8)

    # Logging parameters
    output_dir: str = field(default="/models/trl/sft_checkpoints")