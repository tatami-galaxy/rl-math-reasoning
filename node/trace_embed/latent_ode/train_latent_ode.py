import sys
sys.path.append('../../../')

from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import HfArgumentParser

from embed import Embedder, SentenceTransformerEmbedder
from latent_ode import LatentODE, elbo_batch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:

    # data
    dataset_name: str = field(default="Ujan/deepmath_trace_2")
    dataset_split: str = field(default="train")
    embed_type: str = field(default="sentence-transformers")
    embed_model: str = field(default="sentence-transformers/all-MiniLM-L6-v2")
    device: str = field(default="cuda")
    min_segments: int = field(default=10)
    embed_batch_size: int = field(default=256)
    train_batch_size: int = field(default=32)
    segment_by: str = field(default="\n\n")

    # model
    d_proj: int = field(default=128)       # input projection dim
    d_encoder: int = field(default=128)    # encoder hidden dim
    d_z: int = field(default=64)           # latent ODE dim
    d_ode_hidden: int = field(default=128) # ODE MLP hidden dim
    beta: float = field(default=1.0)       # KL weight in ELBO
    
    # training
    n_epochs: int = field(default=50)
    lr: float = field(default=1e-3)
    seed: int = field(default=42)


# ---------------------------------------------------------------------------
# Data loading (PyTorch)
# ---------------------------------------------------------------------------

def split_trace(trace: str, segment_by: str) -> list[str]:
    """Split a trace into non-empty segments on segment_by."""
    return [s.strip() for s in trace.split(segment_by) if s.strip()]


class TraceDataset(Dataset):
    """Each item is a (T_i, D) float32 tensor — one trajectory per trace."""

    def __init__(self, embeddings: list[np.ndarray]):
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.embeddings[idx])  # (T_i, D)


def collate_fn(batch: list[torch.Tensor]):
    """Pad variable-length trajectories within a batch.

    Returns:
        padded  : (B, T_max, D)  — zero-padded trajectories (right padded)
        mask    : (B, T_max)     — True where data is valid
        lengths : (B,)           — actual trajectory lengths
        ts      : (T_max,)       — uniform timestamps [0, 1, ..., T_max-1]
    """
    lengths = torch.tensor([t.shape[0] for t in batch])
    T_max = int(lengths.max().item())
    D = batch[0].shape[1]

    padded = torch.zeros(len(batch), T_max, D)
    mask = torch.zeros(len(batch), T_max, dtype=torch.bool)
    for i, traj in enumerate(batch):
        T = traj.shape[0]
        padded[i, :T] = traj
        mask[i, :T] = True

    ts = torch.arange(T_max, dtype=torch.float32)
    return padded, mask, lengths, ts


def build_dataset(config: Config, embedder: Embedder) -> TraceDataset:
    print(f"Loading {config.dataset_name} ({config.dataset_split})")
    hf_dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    all_segments_lists: list[list[str]] = []
    for row in hf_dataset:
        segments = split_trace(row["trace"], config.segment_by)
        if len(segments) >= config.min_segments:
            all_segments_lists.append(segments)

    print(
        f"Retained {len(all_segments_lists)}/{len(hf_dataset)} traces "
        f"(min_segments={config.min_segments})"
    )

    lengths = [len(s) for s in all_segments_lists]
    flat_segments = [s for segs in all_segments_lists for s in segs]

    print(f"Embedding {len(flat_segments)} segments with {embedder.__class__.__name__}...")
    flat_embeddings = embedder.embed(flat_segments)  # (total_segments, D)

    split_indices = np.cumsum(lengths)[:-1]
    per_trace = np.split(flat_embeddings, split_indices)  # list of (T_i, D)

    return TraceDataset([arr.astype(np.float32) for arr in per_trace])


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def make_train_step(optimizer: optax.GradientTransformation, beta: float):
    @eqx.filter_jit
    def train_step(model, opt_state, padded, mask, ts, keys):
        loss, grads = eqx.filter_value_and_grad(elbo_batch)(
            model, padded, mask, ts, keys, beta
        )
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss

    return train_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser(Config)
    config = parser.parse_args_into_dataclasses()[0]

    # Embedder
    if config.embed_type == "sentence-transformers":
        embedder = SentenceTransformerEmbedder(
            model_name=config.embed_model,
            device=config.device,
            batch_size=config.embed_batch_size,
        )
    else:
        raise NotImplementedError(f"Embedding type: {config.embed_type}")

    # Dataset + DataLoader
    dataset = build_dataset(config, embedder)
    loader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Model
    d_embed = embedder.dim
    key = jax.random.PRNGKey(config.seed)
    key, model_key = jax.random.split(key)
    model = LatentODE(d_embed, config, key=model_key)

    # Optimizer
    optimizer = optax.adam(config.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(optimizer, config.beta)

    # Training loop
    for epoch in range(config.n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for padded, mask, _, ts in loader:
            B = padded.shape[0]
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, B)

            padded_jax = jnp.array(padded.numpy())
            mask_jax = jnp.array(mask.numpy())
            ts_jax = jnp.array(ts.numpy())

            model, opt_state, loss = train_step(
                model, opt_state, padded_jax, mask_jax, ts_jax, keys
            )
            epoch_loss += float(loss)
            n_batches += 1

        print(f"Epoch {epoch + 1}/{config.n_epochs}  loss={epoch_loss / n_batches:.4f}")


if __name__ == "__main__":
    main()
