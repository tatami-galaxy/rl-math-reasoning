sonar, torch needs to be run from conda env sonar. jax code with uv

##### producer

CUDA_VISIBLE_DEVICES=5 python sonar_gen.py

##### consumer
CUDA_VISIBLE_DEVICES=6 uv run python sonar_trace.py --dataset_name Ujan/deepmath_trace_2