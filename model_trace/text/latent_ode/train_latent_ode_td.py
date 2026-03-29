"""
Train latent ODE with time-dependent vector field.

Uses LatentODETD where dz/dt = f(z, t) — the MLP receives [t, z] as input.
Timestamps are normalized to [0, 1] (via linspace) so that t=0.5 always means
"halfway through the trace" regardless of trace length.

Usage:
CUDA_VISIBLE_DEVICES=0 python -m model_trace.text.latent_ode.train_latent_ode_td \\
    --dataset_name Ujan/deepmath_trace_3
CUDA_VISIBLE_DEVICES=0 python -m model_trace.text.latent_ode.train_latent_ode_td \\
    --dataset_name Ujan/deepmath_trace_3 --filter_topic Algebra \\
    --beta 0.1 --max_steps 500 --save_steps 100
"""

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

from .embed import QwenEmbedder
from .latent_ode import LatentODETD, elbo_batch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:

    # data
    dataset_name: str = field(default="Ujan/deepmath_trace_3")
    dataset_split: str = field(default="train")
    embed_model: str = field(default="Qwen/Qwen3-Embedding-0.6B")
    embed_instruction: str = field(default="Embed this mathematical reasoning segment")
    device: str = field(default="cuda")
    min_segments: int = field(default=10)
    filter_topic: str = field(default="")
    embed_batch_size: int = field(default=256)
    segment_by: str = field(default="\n\n")

    # model
    d_proj: int = field(default=128)
    d_encoder: int = field(default=128)
    d_z: int = field(default=64)
    d_ode_hidden: int = field(default=128)
    d_decoder_depth: int = field(default=3)
    beta: float = field(default=1.0)
    output_dir: str = field(default="models/latent_ode_td")

    # training
    seed: int = field(default=42)
    test_size: float = field(default=0.1)
    train_batch_size: int = field(default=32)
    max_steps: int = field(default=100)
    log_steps: int = field(default=10)
    save_steps: int = field(default=0)
    lr: float = field(default=1e-3)
    max_grad_norm: float = field(default=1.0)
    warmup_fraction: float = field(default=0.05)
    lr_end_factor: float = field(default=0.01)
    curriculum_steps: int = field(default=0)
    curriculum_start_frac: float = field(default=0.1)

    # eval
    eval_sampled: bool = field(default=False)

    # marker for analysis scripts to detect this variant
    time_dependent: bool = field(default=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def split_trace(trace: str, segment_by: str) -> list[str]:
    return [s.strip() for s in trace.split(segment_by) if s.strip()]


class TraceDataset(Dataset):
    """Each item is a (T_i, D) float32 tensor — one trajectory per trace."""

    def __init__(self, embeddings: list[np.ndarray], segments: list[list[str]] | None = None):
        self.embeddings = embeddings
        self.segments = segments

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.embeddings[idx])


def collate_fn(batch: list[torch.Tensor]):
    """Pad variable-length trajectories. Timestamps normalized to [0, 1]."""
    lengths = torch.tensor([t.shape[0] for t in batch])
    T_max = int(lengths.max().item())
    D = batch[0].shape[1]

    padded = torch.zeros(len(batch), T_max, D)
    mask = torch.zeros(len(batch), T_max, dtype=torch.bool)
    for i, traj in enumerate(batch):
        T = traj.shape[0]
        padded[i, :T] = traj
        mask[i, :T] = True

    # Normalized timestamps: [0, 1] so t=0.5 means "halfway" for any trace length
    ts = torch.linspace(0.0, 1.0, T_max)
    return padded, mask, lengths, ts


def filter_by_topic(hf_dataset, topic: str):
    def matches(row):
        parts = [p.strip() for p in row["topic"].split("->")]
        return len(parts) >= 2 and parts[1] == topic
    filtered = hf_dataset.filter(matches)
    print(f"Filtered by topic '{topic}': {len(filtered)}/{len(hf_dataset)} traces")
    return filtered


def build_dataset(config: Config, embedder: QwenEmbedder, max_traces: int | None = None) -> TraceDataset:
    print(f"Loading {config.dataset_name} ({config.dataset_split})")
    hf_dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    if config.filter_topic:
        hf_dataset = filter_by_topic(hf_dataset, config.filter_topic)

    all_segments_lists: list[list[str]] = []
    for row in hf_dataset:
        segments = split_trace(row["trace"], config.segment_by)
        if len(segments) >= config.min_segments:
            all_segments_lists.append(segments)

    print(
        f"Retained {len(all_segments_lists)}/{len(hf_dataset)} traces "
        f"(min_segments={config.min_segments})"
    )

    if max_traces is not None and max_traces < len(all_segments_lists):
        indices = np.random.default_rng(0).choice(len(all_segments_lists), max_traces, replace=False)
        all_segments_lists = [all_segments_lists[i] for i in indices]
        print(f"Subsampled to {len(all_segments_lists)} traces (max_traces={max_traces})")

    lengths = [len(s) for s in all_segments_lists]
    flat_segments = [s for segs in all_segments_lists for s in segs]

    print(f"Embedding {len(flat_segments)} segments with QwenEmbedder...")
    flat_embeddings = embedder.embed(flat_segments)  # (total_segments, D)

    split_indices = np.cumsum(lengths)[:-1]
    per_trace = np.split(flat_embeddings, split_indices)

    return TraceDataset(
        [arr.astype(np.float32) for arr in per_trace],
        segments=all_segments_lists,
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def make_train_step(optimizer: optax.GradientTransformation, beta: float):
    @eqx.filter_jit
    def train_step(model, opt_state, padded, mask, ts, keys):
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
def eval_batch(model: LatentODETD, padded: jax.Array, mask: jax.Array, ts: jax.Array) -> jax.Array:
    def single(x, m):
        mu, _ = model.encoder(x, m)
        zs = model.solve(mu, ts)
        x_hat = model.decode(zs)
        return jnp.sum(m[:, None] * (x_hat - x) ** 2) / jnp.sum(m)

    return jnp.mean(jax.vmap(single)(padded, mask))


def evaluate(model: LatentODETD, loader: DataLoader) -> float:
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
def eval_batch_sampled(
    model: LatentODETD, padded: jax.Array, mask: jax.Array, ts: jax.Array, keys: jax.Array
) -> jax.Array:
    def single(x, m, key):
        z0, _, _ = model.encode(x, m, key)
        zs = model.solve(z0, ts)
        x_hat = model.decode(zs)
        return jnp.sum(m[:, None] * (x_hat - x) ** 2) / jnp.sum(m)

    return jnp.mean(jax.vmap(single)(padded, mask, keys))


def evaluate_sampled(model: LatentODETD, loader: DataLoader, rng_key: jax.Array) -> float:
    total_mse, n_batches = 0.0, 0
    for padded, mask, _, ts in loader:
        rng_key, subkey = jax.random.split(rng_key)
        B = padded.shape[0]
        keys = jax.random.split(subkey, B)
        mse = eval_batch_sampled(
            model,
            jnp.array(padded.numpy()),
            jnp.array(mask.numpy()),
            jnp.array(ts.numpy()),
            keys,
        )
        total_mse += float(mse)
        n_batches += 1
    return total_mse / n_batches


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, config, d_embed, step):
    ckpt_dir = os.path.join(config.output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "model.eqx"), model)
    config_dict = vars(config)
    config_dict["d_embed"] = d_embed
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Checkpoint saved to {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser(Config)
    config = parser.parse_args_into_dataclasses()[0]

    # output dir
    config.output_dir = config.output_dir + '/' + config.dataset_name.split("/")[-1]
    if config.filter_topic:
        config.output_dir = config.output_dir + '_' + config.filter_topic

    embedder = QwenEmbedder(
        model_name=config.embed_model,
        device=config.device,
        batch_size=config.embed_batch_size,
        instruction=config.embed_instruction,
    )

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
    model = LatentODETD(d_embed, config, key=model_key)
    print(f"Model: LatentODETD (time-dependent), d_embed={d_embed}")

    # Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=int(config.max_steps * config.warmup_fraction),
        decay_steps=config.max_steps,
        end_value=config.lr * config.lr_end_factor,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(schedule),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(optimizer, config.beta)

    # Training loop
    writer = SummaryWriter(log_dir=config.output_dir)
    global_step = 0
    done = False
    print(f"Starting training for {config.max_steps} steps...")
    while not done:
        for padded, mask, lengths, ts in loader:
            if global_step >= config.max_steps:
                done = True
                break

            B = padded.shape[0]
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, B)

            # Curriculum: truncate trajectories to a growing fraction
            if config.curriculum_steps > 0 and global_step < config.curriculum_steps:
                frac = config.curriculum_start_frac + (
                    (1.0 - config.curriculum_start_frac) * global_step / config.curriculum_steps
                )
                effective_lengths = (lengths.float() * frac).clamp(min=2).long()
                for i in range(B):
                    mask[i, effective_lengths[i]:] = False

            padded_jax = jnp.array(padded.numpy())
            mask_jax = jnp.array(mask.numpy())
            ts_jax = jnp.array(ts.numpy())

            model, opt_state, loss, recon, kl = train_step(
                model, opt_state, padded_jax, mask_jax, ts_jax, keys
            )
            global_step += 1

            batch_loss = float(loss)
            batch_recon = float(recon)
            batch_kl = float(kl)

            if global_step % config.log_steps == 0:
                writer.add_scalar("loss/step", batch_loss, global_step)
                writer.add_scalar("loss/recon_step", batch_recon, global_step)
                writer.add_scalar("loss/kl_step", batch_kl, global_step)
                print(
                    f"  step {global_step}/{config.max_steps}  "
                    f"loss={batch_loss:.4f}  recon={batch_recon:.4f}  kl={batch_kl:.4f}"
                )

            if config.save_steps > 0 and global_step % config.save_steps == 0:
                save_checkpoint(model, config, d_embed, global_step)

    # Evaluation
    print("Evaluating on test set...")
    if config.eval_sampled:
        test_mse = evaluate_sampled(model, test_loader, key)
        writer.add_scalar("mse/test", test_mse, config.max_steps)
        print(f"Test MSE (sampled, z0~q): {test_mse:.6f}")
    else:
        test_mse = evaluate(model, test_loader)
        writer.add_scalar("mse/test", test_mse, config.max_steps)
        print(f"Test MSE (deterministic, z0=mu): {test_mse:.6f}")

    writer.close()
    save_checkpoint(model, config, d_embed, global_step)


if __name__ == "__main__":
    main()
