import sys
sys.path.append('../../../')
from utils import get_root_dir
from dataclasses import dataclass, field
import os
import json


import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformers import HfArgumentParser

from embed import Embedder, SentenceTransformerEmbedder, SonarEmbedder
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
    segment_by: str = field(default="\n\n")

    # model
    d_proj: int = field(default=128)       # input projection dim
    d_encoder: int = field(default=128)    # encoder hidden dim
    d_z: int = field(default=64)           # latent ODE dim
    d_ode_hidden: int = field(default=128) # ODE MLP hidden dim
    beta: float = field(default=1.0)       # KL weight in ELBO
    
    # training
    seed: int = field(default=42)
    test_size: float = field(default=0.1) 
    train_batch_size: int = field(default=32)
    n_epochs: int = field(default=10)
    log_steps: int = field(default=10)
    lr: float = field(default=1e-3)
    n_decode_examples: int = field(default=3)  # SONAR decode examples after eval


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
        # has_aux=True: elbo_batch returns (loss, (recon, kl))
        # grads are computed w.r.t. the first return value (loss) only
        (loss, (recon, kl)), grads = eqx.filter_value_and_grad(
            lambda m, *a: (lambda r: (r[0], (r[1], r[2])))(elbo_batch(m, *a)),
            has_aux=True,
        )(model, padded, mask, ts, keys, beta)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss, recon, kl

    return train_step


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@eqx.filter_jit
def eval_batch(model: LatentODE, padded: jax.Array, mask: jax.Array, ts: jax.Array) -> jax.Array:
    """Reconstruction MSE on a batch using deterministic z0=mu (no sampling)."""
    def single(x, m):
        # Use mu directly — deterministic evaluation, no stochastic sampling
        mu, _ = model.encoder(x, m)
        zs = model.solve(mu, ts)
        x_hat = model.decode(zs)
        return jnp.sum(m[:, None] * (x_hat - x) ** 2) / jnp.sum(m)

    return jnp.mean(jax.vmap(single)(padded, mask))


def evaluate(model: LatentODE, loader: DataLoader) -> float:
    total_mse, n_batches = 0.0, 0
    for padded, mask, _, ts in loader:
        mse = eval_batch(
            model,
            jnp.array(padded.numpy()),
            jnp.array(mask.numpy()),
            jnp.array(ts.numpy()),
        )
        total_mse += float(mse)
        n_batches += 1
    return total_mse / n_batches


@eqx.filter_jit
def reconstruct_batch(model: LatentODE, padded: jax.Array, mask: jax.Array, ts: jax.Array):
    """Reconstruct embeddings through the ODE (deterministic, z0=mu).

    Returns:
        x_hat: (B, T_max, D) reconstructed embeddings
    """
    def single(x, m):
        mu, _ = model.encoder(x, m)
        zs = model.solve(mu, ts)
        return model.decode(zs)

    return jax.vmap(single)(padded, mask)


def decode_examples(
    model: LatentODE,
    embedder: SonarEmbedder,
    loader: DataLoader,
    n_examples: int,
):
    """Reconstruct test trajectories through the ODE and decode with SONAR.

    For each example trace, prints the SONAR-decoded text of the original
    embeddings side-by-side with the ODE-reconstructed embeddings.
    """
    collected = 0
    print(f"\n{'=' * 70}")
    print("SONAR DECODE EXAMPLES (original vs ODE-reconstructed)")
    print("=" * 70)

    for padded, mask, lengths, ts in loader:
        padded_jax = jnp.array(padded.numpy())
        mask_jax = jnp.array(mask.numpy())
        ts_jax = jnp.array(ts.numpy())

        x_hat = reconstruct_batch(model, padded_jax, mask_jax, ts_jax)
        x_hat_np = np.array(x_hat)
        padded_np = padded.numpy()
        lengths_np = lengths.numpy()

        for i in range(len(padded_np)):
            if collected >= n_examples:
                return
            T = int(lengths_np[i])
            orig_emb = padded_np[i, :T]      # (T, D)
            recon_emb = x_hat_np[i, :T]       # (T, D)

            orig_text = embedder.decode(orig_emb.astype(np.float32))
            recon_text = embedder.decode(recon_emb.astype(np.float32))

            collected += 1
            print(f"\n--- Example {collected} ({T} segments) ---")
            for t in range(T):
                print(f"  [{t}] orig : {repr(orig_text[t][:200])}")
                print(f"       recon: {repr(recon_text[t][:200])}")
                print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser(Config)
    config = parser.parse_args_into_dataclasses()[0]

    # get root dir
    root = get_root_dir()
    # output dir
    output_dir = root + "/models/ode/" + config.dataset_name.split("/")[-1]
    print(f"Output dir: {output_dir}")

    # Embedder
    if config.embed_type == "sentence-transformers":
        embedder = SentenceTransformerEmbedder(
            model_name=config.embed_model,
            device=config.device,
            batch_size=config.embed_batch_size,
        )
    elif config.embed_type == "sonar":
        embedder = SonarEmbedder(
            device=config.device,
            batch_size=config.embed_batch_size,
        )
    else:
        raise NotImplementedError(f"Embedding type: {config.embed_type}")

    # Dataset — split into train / test
    dataset = build_dataset(config, embedder)
    n_test = max(1, int(len(dataset) * config.test_size))
    n_train = len(dataset) - n_test
    train_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(config.seed),
    )
    print(f"Train: {n_train} traces  |  Test: {n_test} traces")

    loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
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
    writer = SummaryWriter(log_dir=output_dir)
    global_step = 1
    print("Starting training...")
    for epoch in range(config.n_epochs):
        epoch_loss, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for padded, mask, _, ts in loader:
            B = padded.shape[0]
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, B)

            padded_jax = jnp.array(padded.numpy())
            mask_jax = jnp.array(mask.numpy())
            ts_jax = jnp.array(ts.numpy())

            model, opt_state, loss, recon, kl = train_step(
                model, opt_state, padded_jax, mask_jax, ts_jax, keys
            )
            batch_loss = float(loss)
            batch_recon = float(recon)
            batch_kl = float(kl)
            epoch_loss += batch_loss
            epoch_recon += batch_recon
            epoch_kl += batch_kl
            n_batches += 1

            if global_step % config.log_steps == 0:
                writer.add_scalar("loss/batch", batch_loss, global_step)
                writer.add_scalar("loss/recon_batch", batch_recon, global_step)
                writer.add_scalar("loss/kl_batch", batch_kl, global_step)

            global_step += 1

        epoch_avg_loss = epoch_loss / n_batches
        epoch_avg_recon = epoch_recon / n_batches
        epoch_avg_kl = epoch_kl / n_batches
        writer.add_scalar("loss/epoch", epoch_avg_loss, epoch)
        writer.add_scalar("loss/recon_epoch", epoch_avg_recon, epoch)
        writer.add_scalar("loss/kl_epoch", epoch_avg_kl, epoch)
        print(
            f"Epoch {epoch + 1}/{config.n_epochs}  "
            f"loss={epoch_avg_loss:.4f}  recon={epoch_avg_recon:.4f}  kl={epoch_avg_kl:.4f}"
        )

    # Evaluation on test set
    print("Evaluating on test set...")
    test_mse = evaluate(model, test_loader)
    writer.add_scalar("mse/test", test_mse, config.n_epochs)
    print(f"Test MSE: {test_mse:.6f}")

    # Decode examples with SONAR
    if config.embed_type == "sonar" and config.n_decode_examples > 0:
        decode_examples(model, embedder, test_loader, config.n_decode_examples)

    writer.close()

    # Save model and config
    os.makedirs(output_dir, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(output_dir, "model.eqx"), model)
    config_dict = vars(config)
    config_dict["d_embed"] = d_embed
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
