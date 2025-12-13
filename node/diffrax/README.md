# Thinking Trace Hidden State Extraction

This code extracts hidden states from reasoning models during thinking traces.

## Files

- `thinking_trace_extractor.py`: Main script to download dataset/model and extract hidden states
- `data_loader.py`: Utilities to load and analyze saved hidden states
- `requirements.txt`: Required dependencies

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run extraction:
```bash
python thinking_trace_extractor.py
```

The script will:
- Download the polaris_53k dataset from huggingface
- Download a small reasoning model (Qwen2.5-3B-Instruct)
- Generate responses with thinking traces
- Extract hidden states for tokens between <think> and </think>
- Store hidden states efficiently in HDF5 format

## Storage

Hidden states are saved in `node/diffrax/hidden_states/`:
- One HDF5 file per sample containing all layer states
- Metadata in `batch_metadata.json`

## Data Format

Each saved sample contains:
- Thinking text between <think> and </think>
- Hidden states for each layer (shape: [num_thinking_tokens, hidden_dim])
- Token indices and metadata