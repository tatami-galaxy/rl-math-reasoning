CUDA_VISIBLE_DEVICES=0 python trace_node.py --hf_token hf_token --trace_layer 18 --step_strategy 0.5 0.5 --length_strategy 0.1 1
CUDA_VISIBLE_DEVICES=2 python trace_node.py --trace_layer 18 --total_steps 5000 --step_strategy 0.2 0.8

CUDA_VISIBLE_DEVICES=1 python extract_trace.py --model_name Qwen/Qwen3-4B-Thinking-2507

##### tensor upload

hf upload-large-folder Ujan/Trace-Tensors --repo-type=dataset Trace-Tensors --num-workers=64