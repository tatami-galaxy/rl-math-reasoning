from huggingface_hub import snapshot_download


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


def segment_representation(trace_dir, config):
    pass