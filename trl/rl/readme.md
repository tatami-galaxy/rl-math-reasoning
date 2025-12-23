##### multi gpu
```
CUDA_VISIBLE_DEVICES=4,5 accelerate launch --config_file /home/ujan/.cache/huggingface/accelerate/gpu_2_config.yaml run_grpo.py --model_name Qwen/Qwen3-4B-Base --acc_reward
```

```
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file /home/ujan/.cache/huggingface/accelerate/gpu_2_config.yaml trl_grpo.py --model_name Qwen/Qwen3-4B-Base --tokenizer_name Qwen/Qwen3-4B-Thinking-2507 --acc_reward
```