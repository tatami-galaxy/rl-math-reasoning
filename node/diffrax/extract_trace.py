import sys
sys.path.append('../../')
from typing import List, Dict, Tuple, Optional
import json
import os
from tqdm import tqdm
import re
import numpy as np
import h5py
from safetensors.torch import save_file
import torch

from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
)
from datasets import load_dataset

from extract_trace_config import ExtractTraceConfig
from utils import SYSTEM_PROMPT, REASONING_START, REASONING_END


class ThinkingTraceExtractor:
    def __init__(self, config):
        self.model = None
        self.tokenizer = None
        self.max_new_tokens = config.max_new_tokens


    def format_prompt(self, question):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Problem : {}".format(question)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text


    def load_model(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=config.model_dtype,
            device_map="auto",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def load_dataset(self, config):
        self.dataset = load_dataset(config.dataset_name, split=config.data_split)
        if config.sample:
            self.dataset = self.dataset.select(range(config.num_samples))


    def extract_thinking_trace_tokens(self, generated_ids):
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        start_pattern = r'{}\s*\n'.format(REASONING_START)
        end_pattern = r'\s*\n{}'.format(REASONING_END)

        # find think start and end positions in text
        start_match = re.search(start_pattern, generated_text)
        end_match = re.search(end_pattern, generated_text)

        if start_match and end_match:
            # find think start and end positions in text
            thinking_start_pos = start_match.end()
            thinking_end_pos = end_match.start()

            # get thinking text
            thinking_text = generated_text[thinking_start_pos:thinking_end_pos]
            # get tokens corresponding to thinking text and convert to tensor
            thinking_tokens = self.tokenizer.encode(thinking_text, add_special_tokens=False)
            thinking_tensor = torch.tensor(thinking_tokens, device=self.model.device).unsqueeze(0)

            # find start and end token position for thinking trace
            start_idx_in_generated = len(self.tokenizer.encode(
                generated_text[:thinking_start_pos], add_special_tokens=False
            ))
            end_idx_in_generated = start_idx_in_generated + len(thinking_tokens)

            return thinking_tensor, torch.tensor([start_idx_in_generated, end_idx_in_generated])

        return None, None


    def generate_with_hidden_states(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            # hidden_states :
            # Outer Tuple (Length = Number of Generated Tokens) : Each element corresponds to one "step" of generation
            # Middle Tuple (Length = Number of Layers + 1)
            # Inner Tensor: The actual hidden state tensor for that specific layer at that specific step
            # shape of hidden_states[step][layer] : 
            # Step 0 (First Token): The shape is (batch_size, prompt_length, hidden_size)
            # Steps 1 to N: The shape is (batch_size, 1, hidden_size)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True
            )     
        # get thinking tokens, start and end indices in output
        # sequences : (batch_size, sequence_length)
        thinking_tokens, thinking_indices = self.extract_thinking_trace_tokens(outputs.sequences)

        # get hidden states
        if thinking_tokens is not None:
            start_idx, end_idx = thinking_indices

            # states for each layer
            thinking_hidden_states = {}
            # token_posn ranges over generation length
            # posn_hidden_states contains layer+1 hidden states for each posn
            for token_posn, posn_hidden_states in enumerate(outputs.hidden_states):
                # TODO : 
                layer_thinking_states = []
                # first position has shape batch_size, prompt_length, hidden_size
                for token_idx in range(start_idx, min(end_idx, len(layer_hidden_states[0]))):
                    layer_thinking_states.append(layer_hidden_states[0][token_idx].cpu().numpy())
                thinking_hidden_states.append(np.array(layer_thinking_states))
            quit()

            return {
                "generated_text": self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False),
                "thinking_text": self.tokenizer.decode(thinking_tokens[0], skip_special_tokens=True),
                "thinking_hidden_states": thinking_hidden_states,
                "thinking_token_indices": thinking_indices.tolist(),
                "num_layers": len(thinking_hidden_states)
            }
        else:
            return {
                "generated_text": self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False),
                "thinking_text": "",
                "thinking_hidden_states": [],
                "thinking_token_indices": [],
                "num_layers": 0
            }


class HiddenStateStorage:
    def __init__(self, storage_dir: str = "node/diffrax/hidden_states"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save_hidden_states_hdf5(self, data: Dict, sample_id: int):
        filepath = os.path.join(self.storage_dir, f"sample_{sample_id:06d}.h5")

        with h5py.File(filepath, 'w') as hf:
            hf.attrs['sample_id'] = sample_id
            hf.attrs['model_name'] = data.get('model_name', '')
            hf.attrs['thinking_text'] = data.get('thinking_text', '')
            hf.attrs['generated_text'] = data.get('generated_text', '')
            hf.attrs['num_layers'] = data.get('num_layers', 0)
            hf.attrs['thinking_token_indices'] = data.get('thinking_token_indices', [])

            thinking_hidden_states = data.get('thinking_hidden_states', [])
            for layer_idx, layer_states in enumerate(thinking_hidden_states):
                if len(layer_states) > 0:
                    hf.create_dataset(f'layer_{layer_idx}', data=layer_states)

    def save_batch_metadata(self, metadata_list: List[Dict]):
        metadata_path = os.path.join(self.storage_dir, "batch_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)


def main(config):
    extractor = ThinkingTraceExtractor(config)
    extractor.load_model(config)
    extractor.load_dataset(config)
    # TODO
    storage = HiddenStateStorage()

    metadata_list = []
    bar = tqdm(total=len(extractor.dataset))
    for i, example in enumerate(extractor.dataset):

        question = example['problem']
        answer = example['answer']
        difficulty = example['difficulty']

        # get prompt
        prompt = extractor.format_prompt(question)

        # TODO
        result = extractor.generate_with_hidden_states(prompt)

        sample_id = i + j
        result['model_name'] = extractor.model_name
        result['sample_id'] = sample_id
        result['question'] = question

        storage.save_hidden_states_hdf5(result, sample_id)

        metadata = {
            'sample_id': sample_id,
            'question': question,
            'thinking_text': result['thinking_text'],
            'num_layers': result['num_layers'],
            'thinking_token_count': len(result['thinking_hidden_states'][0]) if result['thinking_hidden_states'] else 0,
            'has_thinking_trace': len(result['thinking_text']) > 0
        }
        metadata_list.append(metadata)

    storage.save_batch_metadata(metadata_list)

    print(f"Processed {len(metadata_list)} samples. Hidden states saved to {storage.storage_dir}")


if __name__ == "__main__":
    # get config
    parser = HfArgumentParser(ExtractTraceConfig)
    config = parser.parse_args_into_dataclasses()[0]
    main(config)