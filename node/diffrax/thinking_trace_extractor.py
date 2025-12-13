import torch
import numpy as np
import h5py
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import json
import os
from tqdm import tqdm
import re


class ThinkingTraceExtractor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.thinking_start_token = "<think>"
        self.thinking_end_token = "</think>"

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_dataset(self, dataset_name: str = "hamishivi/polaris_53k", split: str = "train"):
        dataset = load_dataset(dataset_name, split=split)
        return dataset

    def extract_thinking_trace_tokens(self, generated_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        start_pattern = r'<think>\s*\n'
        end_pattern = r'\s*\n</think>'

        start_match = re.search(start_pattern, generated_text)
        end_match = re.search(end_pattern, generated_text)

        if start_match and end_match:
            thinking_start_pos = start_match.end()
            thinking_end_pos = end_match.start()

            thinking_text = generated_text[thinking_start_pos:thinking_end_pos]

            thinking_tokens = self.tokenizer.encode(thinking_text, add_special_tokens=False)
            thinking_tensor = torch.tensor(thinking_tokens, device=self.device).unsqueeze(0)

            start_idx_in_generated = len(self.tokenizer.encode(
                generated_text[:thinking_start_pos], add_special_tokens=False
            ))
            end_idx_in_generated = start_idx_in_generated + len(thinking_tokens)

            return thinking_tensor, torch.tensor([start_idx_in_generated, end_idx_in_generated])

        return None, None

    def generate_with_hidden_states(self, prompt: str, max_new_tokens: int = 512) -> Dict:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        thinking_tokens, thinking_indices = self.extract_thinking_trace_tokens(outputs.sequences)

        if thinking_tokens is not None:
            start_idx, end_idx = thinking_indices

            thinking_hidden_states = []
            for layer_idx, layer_hidden_states in enumerate(outputs.hidden_states):
                layer_thinking_states = []
                for token_idx in range(start_idx, min(end_idx, len(layer_hidden_states[0]))):
                    layer_thinking_states.append(layer_hidden_states[0][token_idx].cpu().numpy())
                thinking_hidden_states.append(np.array(layer_thinking_states))

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


def main():
    extractor = ThinkingTraceExtractor()
    extractor.load_model()

    dataset = extractor.load_dataset()
    storage = HiddenStateStorage()

    prompt_template = "Please reason step by step, and put your final answer within \\boxed{}. {question}"

    batch_size = 10
    num_samples = min(100, len(dataset))

    metadata_list = []

    for i in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
        batch_data = dataset[i:i+batch_size]

        for j, sample in enumerate(batch_data):
            question = sample.get('question', sample.get('prompt', ''))
            if not question:
                continue

            prompt = prompt_template.format(question=question)

            try:
                result = extractor.generate_with_hidden_states(prompt, max_new_tokens=512)

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

            except Exception as e:
                print(f"Error processing sample {i+j}: {e}")
                continue

        storage.save_batch_metadata(metadata_list)

    print(f"Processed {len(metadata_list)} samples. Hidden states saved to {storage.storage_dir}")


if __name__ == "__main__":
    main()