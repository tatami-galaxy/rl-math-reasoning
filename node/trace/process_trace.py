import os
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open


def download_layer_traces(root, config):
    layer_pattern = "layer_"+str(config.trace_layer)+"/*"
    trace_dir = root+config.trace_dir+config.trace_data.split('/')[-1]
    snapshot_download(
        repo_id=config.trace_data,
        allow_patterns=layer_pattern,
        local_dir=trace_dir,
        token = config.hf_token,
    )
    return trace_dir


def segment_representations(trace_dir, config):

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    layer_data_dir = trace_dir+"/layer_"+str(config.trace_layer)

    # iterate over tensors from all examples
    for rep_file in os.scandir(layer_data_dir):
        with safe_open(layer_data_dir+"/1.safetensors", framework="jax", device="cpu") as f:
            metadata = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys()}

            # encode thinking text with offset mapping to identify segment boundaries in representations
            enc = tokenizer(metadata['thinking_text'], return_offsets_mapping=True, add_special_tokens=False)
            input_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]

            # sanity check
            assert len(input_ids) == tensors['layer_trace'].shape[0]

            # find segment boundaries
            seg_posns = [i for i, c in enumerate(metadata['thinking_text']) if c == config.segment_by]

            # 
            for rep_id, (start, end) in zip(input_ids, offsets):