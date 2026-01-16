import sys
sys.path.append('../../')
from tqdm import tqdm
import re
import os

from safetensors.torch import save_file
import torch

from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
)
from datasets import load_dataset

from extract_trace_config import ExtractTraceConfig
from utils import(
    get_root_dir,
    SYSTEM_PROMPT,
)


class ThinkingTraceExtractor:
    def __init__(self, config):
        self.model = None
        self.num_layers = None
        self.tokenizer = None
        self.max_new_tokens = config.max_new_tokens
        self.storage_dir = None


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
        self.num_layers = self.model.config.num_hidden_layers


    def load_dataset(self, config):
        self.dataset = load_dataset(config.dataset_name, split=config.data_split)
        if config.sample:
            self.dataset = self.dataset.select(range(config.num_samples))


    def create_storage_dir(self):
        root = get_root_dir()
        model_name = config.model_name.split('/')[-1]
        dataset_name = config.dataset_name.split('/')[-1]
        filename = root+config.data_dir + "/" + dataset_name + "-" + model_name 
        os.makedirs(filename, exist_ok=True)
        for l in range(self.num_layers+1):
            os.makedirs(filename + f"/layer_{l}", exist_ok=True)
        self.storage_dir = filename


    def get_trace_ckpt(self):
        index = 0
        if os.listdir(self.storage_dir + '/layer_0'):
            trace_files = os.listdir(self.storage_dir + '/layer_0')
            # i.safetensors -> get last
            indices = sorted([int(f.split('.')[0]) for f in trace_files])
            index = indices[-1]
        return index


    def extract_thinking_trace_tokens(self, generated_ids):
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        start_pattern = r"<think>"
        end_pattern = r"</think>"

        # find think start and end positions in text
        start_match = re.search(start_pattern, generated_text)
        end_match = re.search(end_pattern, generated_text)

        if start_match and end_match:
            # find think start and end positions in text
            thinking_start_pos = start_match.end() # inclusive
            thinking_end_pos = end_match.start() # not inclusive

            # get thinking text
            thinking_text = generated_text[thinking_start_pos:thinking_end_pos]
            # get tokens corresponding to thinking text and convert to tensor
            thinking_tokens = self.tokenizer.encode(thinking_text, add_special_tokens=False)
            thinking_tensor = torch.tensor(thinking_tokens, device=self.model.device).unsqueeze(0)

            # find start and end token position for thinking trace
            # tokenizing upto <think> separately -> possible num tokens mismatch?
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
            # first position has shape batch_size, prompt_length, hidden_size
            # left shift thinking indices by prompt length
            prompt_length = outputs.hidden_states[0][0].shape[1]
            start_idx = start_idx - prompt_length
            end_idx = end_idx - prompt_length
            # adjust if prompt length off by ~1
            if start_idx < 0:
                diff = -start_idx
                start_idx += diff
                end_idx += diff

            # states for each layer
            # each model -> folder -> folder for each layer -> tensors of traces
            num_layers = len(outputs.hidden_states[0]) # including embedding
            thinking_hidden_states = {l: [] for l in range(num_layers)}
            # token_posn ranges over generation length
            # posn_hidden_states contains layer+1 hidden states for each posn
            # first posn is prompt -> ignore
            for token_posn, posn_hidden_states in enumerate(outputs.hidden_states[1:]):
                if token_posn >= start_idx and token_posn < end_idx:
                    for l in range(num_layers):
                        thinking_hidden_states[l].append(posn_hidden_states[l].flatten().cpu())
                        
            # sanity check
            assert thinking_tokens.shape[1] == len(thinking_hidden_states[0])

            return {
                "generated_text": self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False),
                "thinking_text": self.tokenizer.decode(thinking_tokens[0], skip_special_tokens=True),
                "all_layer_thinking_states": thinking_hidden_states,
                "thinking_token_indices": thinking_indices,
            }
        else:
            return {
                "generated_text": self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False),
                "thinking_text": "",
                "all_layer_thinking_states": [],
                "thinking_token_indices": [],
            }
        

    def save_hidden_states(self, result):
        # generated_text, thinking_text
        # all_layer_thinking_states, thinking_token_indices
        # index, prompt, answer, level
        # separate out layers
        for l, hidden_states in result['all_layer_thinking_states'].items():
            tensor = {"layer_trace" : torch.stack(hidden_states)}
            metadata = {
                "prompt": result["prompt"],
                "answer": result["answer"],
                "level": result["level"],
                "thinking_text": result["thinking_text"],
            }
            filename = "layer_" + str(l) + "/" + result["index"] + ".safetensors"
            save_file(
                tensor,
                self.storage_dir + "/" + filename,
                metadata=metadata
            )


def main(config):

    extractor = ThinkingTraceExtractor(config)
    extractor.load_model(config)
    extractor.load_dataset(config)
    extractor.create_storage_dir()

    # check for already extracted traces
    index = extractor.get_trace_ckpt()

    bar = tqdm(total=len(extractor.dataset))
    fail_count = 0
    for i, example in enumerate(extractor.dataset):

        # skip till ckpt
        if i <= index : 
            bar.update(1)
            continue

        # get prompt
        prompt = extractor.format_prompt(example['problem'])

        # generated_text, thinking_text, all_layer_thinking_states, thinking_token_indices
        result = extractor.generate_with_hidden_states(prompt)

        # no think trace
        if len(result["all_layer_thinking_states"]) < 1: 
            print('failed')
            fail_count += 1
            bar.update(1)
            continue

        # add metadata
        result['index'] = str(i)
        result['prompt'] = prompt
        result['answer'] = example['answer']
        result['level'] = str(example['level']) 

        # save tensors
        extractor.save_hidden_states(result)

        bar.update(1)

    print('Done.')
    print('Num failed : {}'.format(fail_count))
    

if __name__ == "__main__":
    # get config
    parser = HfArgumentParser(ExtractTraceConfig)
    config = parser.parse_args_into_dataclasses()[0]
    main(config)