"""
Reconstruction quality analysis for a trained Latent ODE model.

Computes per-step MSE and cosine similarity between original and
reconstructed embeddings, and plots error-vs-timestep curves.

Usage:
    python -m model_trace.text.latent_ode.analysis.recon_quality \
        --model_dir models/latent_ode/sentence-transformers/deepmath_trace_3 \
        --eval_sampled
"""

import os
import json
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ..latent_ode import LatentODE
from ..train_latent_ode import (
    Config,
    build_dataset,
    collate_fn,
    EmbeddingNormalizer,
)
from ..embed import SentenceTransformerEmbedder, SonarEmbedder


# ---------------------------------------------------------------------------
# Load model (shared with latent_ode_pca — could be factored out later)
# ---------------------------------------------------------------------------

def load_model(model_dir: str) -> tuple[LatentODE, Config, int, EmbeddingNormalizer | None]:
    with open(os.path.join(model_dir, "config.json")) as f:
        config_dict = json.load(f)

    d_embed = config_dict.pop("d_embed")
    config = Config(**config_dict)

    skeleton = LatentODE(d_embed, config, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(
        os.path.join(model_dir, "model.eqx"), skeleton
    )

    normalizer_path = os.path.join(model_dir, "normalizer.npz")
    normalizer = EmbeddingNormalizer.load(normalizer_path) if os.path.exists(normalizer_path) else None
    if normalizer is not None:
        print("Loaded normalizer from", normalizer_path)

    return model, config, d_embed, normalizer


# ---------------------------------------------------------------------------
# Reconstruction: collect per-step errors
# ---------------------------------------------------------------------------

@eqx.filter_jit
def _reconstruct_deterministic(model: LatentODE, padded, mask, ts):
    def single(x, m):
        mu, _ = model.encoder(x, m)
        zs = model.solve(mu, ts)
        return model.decode(zs)
    return jax.vmap(single)(padded, mask)


@eqx.filter_jit
def _reconstruct_sampled(model: LatentODE, padded, mask, ts, keys):
    def single(x, m, key):
        z0, _, _ = model.encode(x, m, key)
        zs = model.solve(z0, ts)
        return model.decode(zs)
    return jax.vmap(single)(padded, mask, keys)


def collect_per_step_errors(
    model: LatentODE,
    loader: DataLoader,
    sampled: bool = False,
    rng_key: jax.Array | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Compute per-step MSE and cosine similarity for every trajectory.

    Returns three lists (one entry per trajectory, variable length):
        mse_steps:  list of (T_i,) arrays — MSE at each step
        cos_steps:  list of (T_i,) arrays — cosine similarity at each step
        lengths:    list of ints — trajectory lengths
    """
    mse_all, cos_all, lengths_all = [], [], []

    for padded, mask, lengths, ts in loader:
        padded_jax = jnp.array(padded.numpy())
        mask_jax = jnp.array(mask.numpy())
        ts_jax = jnp.array(ts.numpy())

        if sampled:
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, padded.shape[0])
            x_hat = _reconstruct_sampled(model, padded_jax, mask_jax, ts_jax, keys)
        else:
            x_hat = _reconstruct_deterministic(model, padded_jax, mask_jax, ts_jax)

        x_hat_np = np.array(x_hat)
        padded_np = padded.numpy()
        lengths_np = lengths.numpy()

        for i in range(len(padded_np)):
            T = int(lengths_np[i])
            orig = padded_np[i, :T]       # (T, D)
            recon = x_hat_np[i, :T]       # (T, D)

            # per-step MSE: mean over embedding dims
            mse = np.mean((orig - recon) ** 2, axis=1)  # (T,)

            # per-step cosine similarity
            dot = np.sum(orig * recon, axis=1)
            norm_orig = np.linalg.norm(orig, axis=1).clip(1e-8)
            norm_recon = np.linalg.norm(recon, axis=1).clip(1e-8)
            cos = dot / (norm_orig * norm_recon)  # (T,)

            mse_all.append(mse)
            cos_all.append(cos)
            lengths_all.append(T)

    return mse_all, cos_all, lengths_all


# ---------------------------------------------------------------------------
# Aggregate into error-vs-timestep
# ---------------------------------------------------------------------------

def aggregate_by_timestep(
    per_step_values: list[np.ndarray],
    lengths: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate variable-length per-step values by timestep position.

    Returns:
        positions: (T_max,) timestep indices
        means:     (T_max,) mean value at each position
        stds:      (T_max,) std at each position
    """
    T_max = max(lengths)
    buckets = [[] for _ in range(T_max)]
    for vals, T in zip(per_step_values, lengths):
        for t in range(T):
            buckets[t].append(vals[t])

    positions = np.arange(T_max)
    means = np.array([np.mean(b) for b in buckets])
    stds = np.array([np.std(b) for b in buckets])
    counts = np.array([len(b) for b in buckets])

    return positions, means, stds, counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_error_vs_timestep(
    positions: np.ndarray,
    mse_means: np.ndarray,
    mse_stds: np.ndarray,
    cos_means: np.ndarray,
    cos_stds: np.ndarray,
    counts: np.ndarray,
    save_path: str | None = None,
):
    # only plot positions with enough samples
    min_samples = 5
    valid = counts >= min_samples

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MSE vs timestep
    ax = axes[0]
    ax.plot(positions[valid], mse_means[valid], color="steelblue", linewidth=1.5)
    ax.fill_between(
        positions[valid],
        (mse_means - mse_stds)[valid],
        (mse_means + mse_stds)[valid],
        alpha=0.25, color="steelblue",
    )
    ax.set_xlabel("Timestep (segment position)")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction MSE vs. timestep")
    ax.grid(True, alpha=0.3)

    # Cosine similarity vs timestep
    ax = axes[1]
    ax.plot(positions[valid], cos_means[valid], color="coral", linewidth=1.5)
    ax.fill_between(
        positions[valid],
        (cos_means - cos_stds)[valid],
        (cos_means + cos_stds)[valid],
        alpha=0.25, color="coral",
    )
    ax.set_xlabel("Timestep (segment position)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Reconstruction cosine similarity vs. timestep")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reconstruction quality analysis")
    parser.add_argument("--model_dir", required=True, help="Directory containing model.eqx and config.json")
    parser.add_argument("--dataset_name", required=True, help="HuggingFace dataset name (e.g. Ujan/deepmath_trace_3)")
    parser.add_argument("--dataset_split", default="train", help="Dataset split to evaluate on")
    parser.add_argument("--eval_sampled", action="store_true", help="Use sampled z0 instead of deterministic mu")
    parser.add_argument("--max_traces", type=int, default=500, help="Max traces to evaluate")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_dir}")
    model, config, d_embed, normalizer = load_model(args.model_dir)

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

    # Override dataset from CLI args
    config.dataset_name = args.dataset_name
    config.dataset_split = args.dataset_split

    # Dataset
    dataset = build_dataset(config, embedder)

    if normalizer is not None:
        print("Applying normalizer to embeddings...")
        for i in range(len(dataset.embeddings)):
            dataset.embeddings[i] = normalizer.normalize(dataset.embeddings[i]).astype(np.float32)

    if args.max_traces < len(dataset):
        import torch
        indices = np.random.default_rng(0).choice(len(dataset), args.max_traces, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    loader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=False, collate_fn=collate_fn,
    )

    # Collect per-step errors
    mode = "sampled z0~q" if args.eval_sampled else "deterministic z0=mu"
    print(f"Computing reconstruction errors ({mode})...")
    rng_key = jax.random.PRNGKey(config.seed) if args.eval_sampled else None
    mse_steps, cos_steps, lengths = collect_per_step_errors(
        model, loader, sampled=args.eval_sampled, rng_key=rng_key,
    )

    # Global summary
    all_mse = np.concatenate(mse_steps)
    all_cos = np.concatenate(cos_steps)
    print(f"\n--- Global reconstruction summary ({len(lengths)} traces) ---")
    print(f"  MSE    : mean={all_mse.mean():.6f}  std={all_mse.std():.6f}  "
          f"median={np.median(all_mse):.6f}")
    print(f"  Cosine : mean={all_cos.mean():.4f}  std={all_cos.std():.4f}  "
          f"median={np.median(all_cos):.4f}")

    # Aggregate by timestep
    pos, mse_means, mse_stds, counts = aggregate_by_timestep(mse_steps, lengths)
    _, cos_means, cos_stds, _ = aggregate_by_timestep(cos_steps, lengths)

    # Print a table for the first 20 positions
    print(f"\n--- Per-timestep breakdown (first 20) ---")
    print(f"  {'t':>4s}  {'MSE':>10s}  {'Cosine':>10s}  {'n_traces':>8s}")
    for t in range(min(20, len(pos))):
        print(f"  {t:4d}  {mse_means[t]:10.6f}  {cos_means[t]:10.4f}  {counts[t]:8d}")

    # Plot
    save_path = os.path.join(args.model_dir, "recon_quality.png")
    plot_error_vs_timestep(pos, mse_means, mse_stds, cos_means, cos_stds, counts, save_path)


if __name__ == "__main__":
    main()
