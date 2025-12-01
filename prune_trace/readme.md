CUDA_VISIBLE_DEVICES=0,1 python prune_trace.py --dataset_name Ujan/DeepMath-103K_samples_10000_seq_16384 --segment_length 128 --sparsity 0.5

CUDA_VISIBLE_DEVICES=1,2,3,4 python prune_trace.py --dataset_name Ujan/DeepMath-103K_samples_10000_seq_16384 --model_name Qwen/Qwen3-30B-A3B-Thinking-2507 --layers 17,18,19