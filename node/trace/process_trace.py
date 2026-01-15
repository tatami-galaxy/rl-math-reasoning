import os
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

import torch

from tqdm.auto import tqdm


def download_layer_traces(root, config):
    if config.hf_token is None: raise ValueError("Pass in HF token")
    layer_pattern = "layer_"+str(config.trace_layer)+"/*"
    trace_dir = root+config.trace_dir+config.trace_data.split('/')[-1]
    snapshot_download(
        repo_id=config.trace_data,
        allow_patterns=layer_pattern,
        local_dir=trace_dir,
        token = config.hf_token,
    )
    return trace_dir


def load_segment_representations(root, config):

    # filename
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    layer_data_dir = root + "/" + config.trace_dir
    layer_data_dir += "/" if not layer_data_dir.endswith("/") else ""
    layer_data_dir += "/" + config.dataset_name.split("/")[-1] + "-" + config.model_name.split("/")[-1]
    layer_data_dir += "/" + "layer_" + str(config.trace_layer)
    layer_segment_dir = layer_data_dir + "_segments"

    # get seg reps for each example
    seg_reps = []
    meta = []
    for rep_file in os.scandir(layer_segment_dir):
        if not rep_file.name.endswith(".safetensors"): continue
        with safe_open(layer_segment_dir+"/"+rep_file.name, framework="jax", device="cpu") as f:
            # TODO : get metadata here
            metadata = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys()}
            seg_reps.append(tensors['layer_trace'])
            meta.append(metadata)
            
    return seg_reps, meta
    



def segment_representations(root, config):

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    layer_data_dir = root + "/" + config.trace_dir
    layer_data_dir += "/" if not layer_data_dir.endswith("/") else ""
    layer_data_dir += "/" + config.dataset_name.split("/")[-1] + "-" + config.model_name.split("/")[-1]
    layer_data_dir += "/" + "layer_" + str(config.trace_layer)

    # layer segments directory
    layer_segment_dir = layer_data_dir + "_segments"
    if os.path.isdir(layer_segment_dir) and os.listdir(layer_segment_dir) and not config.resegment:
        print("Segmentation already done")
        return
    else:
        print("Segmenting...")
        os.makedirs(layer_segment_dir, exist_ok=True)

    # iterate over tensors from all examples and save segment reps for each example
    bar = tqdm(range((len(os.listdir(layer_data_dir)))))
    for rep_file in os.scandir(layer_data_dir):
        if not rep_file.name.endswith(".safetensors"): continue
        with safe_open(layer_data_dir+"/"+rep_file.name, framework="pt", device="cpu") as f:

            metadata = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys()}

            # encode thinking text with offset mapping to identify segment boundaries in representations
            # return_offsets_mapping -> return (char_start, char_end) for each token
            enc = tokenizer(metadata['thinking_text'], return_offsets_mapping=True, add_special_tokens=False)
            input_ids = enc["input_ids"] # token ids
            offsets = enc["offset_mapping"]

            # sanity check
            assert len(input_ids) == tensors['layer_trace'].shape[0]
            assert len(offsets) == tensors['layer_trace'].shape[0]

            # find segment boundaries
            seg_posns = [i for i, c in enumerate(metadata['thinking_text']) if c == config.segment_by]
            # process segment lengths
            # merge empty segments
            new_seg_posns = []
            for i in range(len(seg_posns)-1):
                if seg_posns[i+1] - seg_posns[i] > 1:
                    new_seg_posns.append(seg_posns[i])
            new_seg_posns.append(seg_posns[-1])
            seg_posns = new_seg_posns
            if seg_posns[0] == 0: seg_posns = seg_posns[1:]

            # skip example if very few segments
            if len(seg_posns) < (config.min_segments-1): continue

            # collect token reps for each segment and average
            segment_reps = []
            current_seg = []
            seg_id = 0
            for token_rep, (start, end) in zip(tensors['layer_trace'], offsets):
                if seg_id < len(seg_posns) and start >= seg_posns[seg_id]:
                    avg_seg_rep = torch.mean(torch.stack(current_seg), dim=0)
                    segment_reps.append(avg_seg_rep)
                    current_seg = []
                    seg_id += 1
                current_seg.append(token_rep)

            # save segment reps
            filename = layer_segment_dir + "/" + rep_file.name
            tensor = {"layer_trace" : torch.stack(segment_reps)}
            save_file(
                tensor,
                filename,
                metadata=metadata
            )

        bar.update(1)

