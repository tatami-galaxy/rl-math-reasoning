import h5py
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
import torch


class HiddenStateLoader:
    def __init__(self, storage_dir: str = "node/diffrax/hidden_states"):
        self.storage_dir = storage_dir
        self.metadata_path = os.path.join(storage_dir, "batch_metadata.json")
        self.metadata = None
        self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

    def load_hidden_states(self, sample_id: int) -> Optional[Dict]:
        filepath = os.path.join(self.storage_dir, f"sample_{sample_id:06d}.h5")

        if not os.path.exists(filepath):
            return None

        data = {}
        with h5py.File(filepath, 'r') as hf:
            data['sample_id'] = hf.attrs['sample_id']
            data['model_name'] = hf.attrs.get('model_name', '')
            data['thinking_text'] = hf.attrs.get('thinking_text', '')
            data['generated_text'] = hf.attrs.get('generated_text', '')
            data['num_layers'] = hf.attrs.get('num_layers', 0)
            data['thinking_token_indices'] = list(hf.attrs.get('thinking_token_indices', []))

            data['thinking_hidden_states'] = []
            for layer_idx in range(data['num_layers']):
                layer_key = f'layer_{layer_idx}'
                if layer_key in hf:
                    data['thinking_hidden_states'].append(np.array(hf[layer_key]))

        return data

    def load_layer_states(self, sample_id: int, layer_idx: int) -> Optional[np.ndarray]:
        filepath = os.path.join(self.storage_dir, f"sample_{sample_id:06d}.h5")

        if not os.path.exists(filepath):
            return None

        with h5py.File(filepath, 'r') as hf:
            layer_key = f'layer_{layer_idx}'
            if layer_key in hf:
                return np.array(hf[layer_key])

        return None

    def get_samples_with_thinking(self) -> List[Dict]:
        return [sample for sample in self.metadata if sample.get('has_thinking_trace', False)]

    def get_dataset_statistics(self) -> Dict:
        if not self.metadata:
            return {}

        stats = {
            'total_samples': len(self.metadata),
            'samples_with_thinking': len(self.get_samples_with_thinking()),
            'average_thinking_tokens': 0,
            'layer_statistics': {}
        }

        thinking_samples = self.get_samples_with_thinking()
        if thinking_samples:
            stats['average_thinking_tokens'] = np.mean([s['thinking_token_count'] for s in thinking_samples])

        sample_path = os.path.join(self.storage_dir, f"sample_{self.metadata[0]['sample_id']:06d}.h5")
        if os.path.exists(sample_path):
            with h5py.File(sample_path, 'r') as hf:
                num_layers = hf.attrs.get('num_layers', 0)
                stats['layer_statistics'] = {
                    'num_layers': num_layers,
                    'hidden_dim': None
                }

                if num_layers > 0 and 'layer_0' in hf:
                    layer_0_shape = hf['layer_0'].shape
                    if len(layer_0_shape) >= 2:
                        stats['layer_statistics']['hidden_dim'] = layer_0_shape[-1]

        return stats


class HiddenStateDataset(torch.utils.data.Dataset):
    def __init__(self, storage_dir: str = "node/diffrax/hidden_states",
                 only_thinking: bool = True,
                 layer_indices: Optional[List[int]] = None):
        self.loader = HiddenStateLoader(storage_dir)
        self.only_thinking = only_thinking
        self.layer_indices = layer_indices

        if only_thinking:
            self.samples = self.loader.get_samples_with_thinking()
        else:
            self.samples = self.loader.metadata

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.samples[idx]['sample_id']
        data = self.loader.load_hidden_states(sample_id)

        if not data:
            return None

        hidden_states = data['thinking_hidden_states']

        if self.layer_indices is not None:
            hidden_states = [hidden_states[i] for i in self.layer_indices if i < len(hidden_states)]

        return {
            'sample_id': sample_id,
            'question': data.get('question', ''),
            'thinking_text': data['thinking_text'],
            'hidden_states': hidden_states,
            'num_layers': len(hidden_states)
        }