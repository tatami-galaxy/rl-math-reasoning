##### multi gpu
```
CUDA_VISIBLE_DEVICES=4,5 accelerate launch --config_file /home/ujan/.cache/huggingface/accelerate/gpu_2_config.yaml run_grpo.py --model_name Qwen/Qwen3-4B-Base --acc_reward
```

```
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file /home/ujan/.cache/huggingface/accelerate/gpu_2_config.yaml trl_grpo.py --model_name Ujan/Qwen3-4B-Base_DeepMath-103K_samples_10000_seq_4096_epoch_1 --tokenizer_name Qwen/Qwen3-4B-Thinking-2507 --use_peft --acc_reward --max_gen_len 8192
```