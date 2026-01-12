CUDA_VISIBLE_DEVICES=0 python trace_node.py --hf_token hf_token --trace_layer 18

CUDA_VISIBLE_DEVICES=1 python extract_trace.py --model_name Qwen/Qwen3-4B-Thinking-2507