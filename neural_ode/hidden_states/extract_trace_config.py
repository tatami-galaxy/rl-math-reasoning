from dataclasses import dataclass, field


@dataclass
class ExtractTraceConfig:

    # seed
    seed: int = 42

    # model
    model_name: str = field(default="Qwen/Qwen3-4B-Thinking-2507")
    model_dtype: str = field(default="auto")
    max_new_tokens: int = field(default=32768)

    # dataset
    dataset_name: str = field(default="HuggingFaceH4/MATH-500")
    data_split: str = field(default="test")
    sample: bool = field(default=False)
    num_samples: int = field(default=1000)

    # storage
    data_dir: str = field(default="/data/trace_tensors/")