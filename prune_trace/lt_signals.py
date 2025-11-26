from utils import REASONING_START, REASONING_END
import torch
from datasets import Dataset
from tqdm.auto import tqdm


class LTSignals:

    def __init__(self, model_args):
        # model
        self.model = model_args.model
        self.tokenizer = model_args.tokenizer
        self.batch_size = model_args.batch_size
        self.segment_length = model_args.segment_length
        self.overlap_segments = model_args.overlap_segments
        self.stride = model_args.stride
        self.layers = model_args.layers
        self.sparsity = model_args.sparsity
        self.pruning_logic = model_args.pruning_logic


    # assumes trace exists, demarcated by REASONING_START and REASONING_END
    def generate_think_trace_mask(self, input_ids):
        think_start_token_id = self.tokenizer.convert_tokens_to_ids(REASONING_START)
        think_end_token_id = self.tokenizer.convert_tokens_to_ids(REASONING_END)
        trace_mask = torch.zeros_like(input_ids)
        # iterate over batch
        trace_start_indices = []
        trace_end_indices = []
        for i in range(input_ids.shape[0]):
            trace_start_index = torch.where(input_ids[i] == think_start_token_id)[0].item()
            trace_end_index = torch.where(input_ids[i] == think_end_token_id)[0].item()
            trace_mask[i, trace_start_index+1 : trace_end_index] = 1
            trace_start_indices.append(trace_start_index+1)
            trace_end_indices.append(trace_end_index)
        return trace_mask, trace_start_indices, trace_end_indices    


    def do_segmentation(self, all_layer_hidden_states, input_ids, current_index, end_index):
        segment_states = []
        segment_texts = []

        # iterate over trace, segment by segment
        while (current_index + self.segment_length) <= end_index:
            # get segment hidden representations
            segment_hidden_states = all_layer_hidden_states[:, current_index: current_index+self.segment_length]
            # get token ids for segment
            segment_ids = input_ids[current_index: current_index+self.segment_length]
            # get text
            segment_text = self.tokenizer.decode(segment_ids)
            # add to lists
            segment_states.append(segment_hidden_states)
            segment_texts.append(segment_text)
            # update index
            if self.overlap_segments:
                current_index = current_index + self.stride
            else:
                current_index = current_index + self.segment_length
        # replace last segment
        if current_index != end_index:
            segment_hidden_states = all_layer_hidden_states[:, end_index-self.segment_length: end_index]
            segment_ids = input_ids[end_index-self.segment_length: end_index]
            segment_text = self.tokenizer.decode(segment_ids)
            segment_states[-1] = segment_hidden_states
            segment_texts[-1] = segment_text 
        # num_segments, num_layers, segment_length, hidden_size
        segment_states = torch.stack(segment_states)
        
        return segment_states, segment_texts
   

    def get_segment_states(
            self,
            think_trace_hidden_states,
            input_ids,
            trace_start_indices,
            trace_end_indices,
        ):
        batch_segment_states = []
        batch_segment_texts = []
        current_batch_size = len(input_ids)

        # iterate over batch
        for b in range(current_batch_size):
            # get all layer hidden states for each example in batch
            # num_layers+1, max_trace_length, hidden_size
            all_layer_hidden_states = torch.stack(
                    [layer_hidden_states[b] for layer_hidden_states in think_trace_hidden_states]
            )
            # inputs ids for current example
            example_input_ids = input_ids[b]
            current_index = trace_start_indices[b]
            end_index = trace_end_indices[b]

            # get segment states and text
            segment_states, segment_texts = self.do_segmentation(
                    all_layer_hidden_states,
                    example_input_ids,
                    current_index,
                    end_index,
                )

            batch_segment_states.append(segment_states)
            batch_segment_texts.append(segment_texts)
        
        return batch_segment_states, batch_segment_texts


    def segment_repr(self, batch):  
        # tokenize batch
        # input_ids, attention_mask
        model_inputs = self.tokenizer(
            batch["text"], 
            padding=True,
            return_tensors="pt").to(self.model.device)
        
        # get think trace mask
        # only start_indices are inclusive
        think_trace_mask, trace_start_indices, trace_end_indices = self.generate_think_trace_mask(
                model_inputs["input_ids"]
        )
        
        # pass through model
        with torch.no_grad():
            # logits, hidden_states (1 + num_layers)
            output = self.model(**model_inputs, output_hidden_states=True)
            # get hidden states 
            hidden_states_all_layers = output["hidden_states"]
            
            # get hidden states for only the think trace
            # list of size 1 + num_layers
            # each item -> b, max_trace_length, hidden_size
            # zeroing out hidden_states beyond the think trace
            think_trace_hidden_states = [h * think_trace_mask.unsqueeze(-1) for h in hidden_states_all_layers]

            # segment hidden states (fixed size)
            # list of length batch_size -> num_segments, num_layers, segment_length, hidden_size 
            batch_segment_states, batch_segment_texts = self.get_segment_states(
                think_trace_hidden_states,
                model_inputs["input_ids"],
                trace_start_indices,
                trace_end_indices,
            )

            # average across segments
            avg_batch_segment_states = [b.mean(dim=2) for b in batch_segment_states]
            
        return avg_batch_segment_states, batch_segment_texts
    

    def net_change(self, avg_batch_segment_states):
        # reasoning drift vectors for all layers
        current_batch_size = len(avg_batch_segment_states)
        batch_reasoning_drift_all_layers = []
        batch_net_change = []

        # iterate over batch
        for b in range(current_batch_size):
            # segment states for all layers in example
            # num_segments, num_layers+1, hidden_size
            avg_segment_states = avg_batch_segment_states[b]
            num_segments = avg_segment_states.shape[0]

            # subtract first reps from final reps
            # num_layers+1, hidden_size
            reasoning_drift_all_layers = avg_segment_states[0,:,:] - avg_segment_states[-1,:,:]

            # net change
            # calc norm
            reasoning_drift_all_layers_norm = torch.linalg.vector_norm(reasoning_drift_all_layers, dim=1)
            # length normalize
            reasoning_drift_all_layers_norm = reasoning_drift_all_layers_norm / num_segments
            # average
            net_change = torch.mean(reasoning_drift_all_layers_norm) 

            batch_reasoning_drift_all_layers.append(reasoning_drift_all_layers)
            batch_net_change.append(net_change)
        
        return torch.stack(batch_reasoning_drift_all_layers), batch_net_change
    

    def cumulative_change(self, avg_batch_segment_states):
        # update vectors for all layers
        current_batch_size = len(avg_batch_segment_states)
        update_vectors_all_layers = []
        cumul_change = []

        # iterate over batch
        for b in range(current_batch_size):
            # roll and subtract to get update vectors
            # num_segments, num_layers+1, hidden_size
            avg_segment_states = avg_batch_segment_states[b]
            avg_segment_states_shift = torch.roll(avg_segment_states, 1, dims=0)
            # make the first segment states same so that substraction = 0
            avg_segment_states_shift[0,:,:] = avg_segment_states[0,:,:]

            # update vectors
            update_vectors = avg_segment_states - avg_segment_states_shift
            update_vectors_all_layers.append(update_vectors)

            # cumulative change
            update_vectors_norm = torch.linalg.vector_norm(update_vectors, dim=-1)
            update_vectors_norm_sum = torch.sum(update_vectors_norm, dim=0)
            cumul_change.append(torch.mean(update_vectors_norm_sum))

        return update_vectors_all_layers, cumul_change


    def aligned_change(self, batch_reasoning_drift_all_layers, batch_update_all_layers):
        current_batch_size = len(batch_reasoning_drift_all_layers)
        cos_sim_all_layers = []
        algn_change = [] 

        # iterate over batch
        for b in range(current_batch_size):
            # reasoning drift vectors for all layers
            reasoning_drift_all_layers = batch_reasoning_drift_all_layers[b]
            # update vectors for all layers
            update_all_layers = batch_update_all_layers[b]
            
            # cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                    reasoning_drift_all_layers, update_all_layers, dim=-1
            )
            cos_sim_all_layers.append(cos_sim)

            # aligned change
            algn_change.append(torch.mean((torch.sum(cos_sim, dim=0)/(cos_sim.shape[0] - 1))))

        return cos_sim_all_layers, algn_change
    

    def get_segments_by_algn(self, batch_cos_sim_all_layers, batch_segment_texts):
        # get batch size
        current_batch_size = len(batch_cos_sim_all_layers)

        # avg over layers
        if self.layers == "all":
            batch_cos_sim_layer_avg = [torch.mean(cos_sim, dim=1) for cos_sim in batch_cos_sim_all_layers]
        else:
            layers = [int(l) for l in self.layers.split(',')]
            batch_cos_sim_layer_avg = [
                    torch.mean(cos_sim[:,layers], dim=1) for cos_sim in batch_cos_sim_all_layers
            ] 

        # sort segments by layer-averaged cos sim
        batch_keep_segments = []
        batch_sorted_cos_ind = []

        # iterate over batch
        for b in range(current_batch_size):
            # ignore first segment
            segments = batch_segment_texts[b][1:]
            cos_sim_layer_avg = batch_cos_sim_layer_avg[b][1:]

            # sort by cos values
            sorted_cos, sorted_cos_ind = torch.sort(cos_sim_layer_avg)

            # get number of segments to keep
            num_keep_segments = round(sorted_cos.shape[0] * self.sparsity)

            # get indices of segments to keep and sort them
            # the first num_keep_segment indices are to be removed
            keep_indices, _ = torch.sort(sorted_cos_ind[num_keep_segments:])

            # select segments to keep
            keep_segments = [segments[k] for k in keep_indices]

            # add back first segment
            keep_segments = [batch_segment_texts[b][0]] + keep_segments
            # add first index and increase remaining indices by 1
            sorted_cos_ind = torch.cat((torch.tensor([0]).to(sorted_cos_ind.device), sorted_cos_ind+1))
            batch_sorted_cos_ind.append(sorted_cos_ind)
            batch_keep_segments.append(keep_segments)

        return batch_sorted_cos_ind, batch_keep_segments
    

    def prune_dataset(self, dataloader):
        all_texts = []
        bar = tqdm(range(len(dataloader)))
        for batch in dataloader:
            # how to segment?
            # 1. fixed segment size
            # 2. find boundary?
            # get segment representations for all layers -> list of batch_size
            # each element -> num_segments, num_layers+1, hidden_size
            # batch_segment_texts -> list of length batch_size, of list of segments
            avg_batch_segment_states, batch_segment_texts = self.segment_repr(batch)

            # net change
            batch_reasoning_drift_all_layers, batch_net_change = self.net_change(avg_batch_segment_states)

            # cumulative change
            # list of batch_size
            # each element -> num_segments, num_layers+1, hidden_size
            batch_update_all_layers, batch_cumul_change = self.cumulative_change(avg_batch_segment_states)

            # aligned change
            # list of batch_size
            # each element -> num_segments, num_layers+1
            # note : first segment values are meaningless
            batch_cos_sim_all_layers, batch_algn_change = self.aligned_change(
                batch_reasoning_drift_all_layers,
                batch_update_all_layers,
            )

            if self.pruning_logic == 'algn':
                batch_sorted_cos_indices, batch_pruned_segments = self.get_segments_by_algn(
                    batch_cos_sim_all_layers,
                    batch_segment_texts,
                )
            else:
                raise NotImplementedError()

            # join segments
            batch_pruned_segments = ['\n\n'.join(segments) for segments in batch_pruned_segments]

            # create examples using pruned trace
            # add pruned trace between intro and outro
            intros = [trace.split(REASONING_START)[0] for trace in batch['text']]
            outros = [trace.split(REASONING_END)[-1] for trace in batch['text']]
            texts = [
                    intros[b]+REASONING_START+batch_pruned_segments[b]+REASONING_END+outros[b] for b in range(
                        len(batch['text'])
                    )
            ]
            
            # create dataset in memory
            all_texts.extend(texts)    

            bar.update(1)

        return Dataset.from_dict({'text': all_texts})