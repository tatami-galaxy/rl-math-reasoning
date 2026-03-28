"""
Nearest-neighbor retrieval analysis for a trained Latent ODE model.

For each reconstructed embedding x_hat[t], find its nearest neighbor in the
original segment embedding corpus. Measures how often the true segment is
rank-1 (or top-k), giving a semantic quality signal without text decoding.

Metrics reported:
  - Recall@1, @5, @10 (global and per-timestep)
  - Mean Reciprocal Rank (MRR)
  - Per-timestep breakdown

Usage:
    python -m model_trace.text.latent_ode.analysis.nn_retrieval \\
        --model_dir models/latent_ode/sentence-transformers/deepmath_trace_3 \\
        --dataset_name Ujan/deepmath_trace_3
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
from .recon_quality import load_model


# ---------------------------------------------------------------------------
# Reconstruction (same as recon_quality — kept local to avoid circular deps)
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


# ---------------------------------------------------------------------------
# Build corpus and compute retrieval ranks
# ---------------------------------------------------------------------------

def build_corpus(dataset) -> np.ndarray:
    """Flatten all trajectory embeddings into a single (N, D) corpus matrix."""
    return np.concatenate(
        [dataset[i].numpy() for i in range(len(dataset))], axis=0,
    )  # (N_total, D)


def compute_retrieval_ranks(
    model: LatentODE,
    loader: DataLoader,
    corpus: np.ndarray,
    corpus_offsets: np.ndarray,
    sampled: bool = False,
    rng_key: jax.Array | None = None,
    metric: str = "cosine",
) -> tuple[list[np.ndarray], list[int]]:
    """Compute the rank of the true segment among corpus neighbors for each reconstruction.

    Args:
        corpus: (N, D) all original segment embeddings.
        corpus_offsets: (num_traces,) cumulative start index of each trace in corpus.
        metric: "cosine" or "l2".

    Returns:
        ranks_all: list of (T_i,) int arrays — rank of true segment (0-indexed)
        lengths_all: list of ints
    """
    # Precompute corpus norms for cosine similarity
    if metric == "cosine":
        corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True).clip(1e-8)  # (N, 1)
        corpus_normed = corpus / corpus_norms  # (N, D)

    ranks_all = []
    lengths_all = []
    trace_idx = 0  # tracks which trace we're on globally

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
        lengths_np = lengths.numpy()

        for i in range(len(lengths_np)):
            T = int(lengths_np[i])
            recon = x_hat_np[i, :T]  # (T, D)

            # Global index of true segments in corpus
            true_start = corpus_offsets[trace_idx]
            true_indices = np.arange(true_start, true_start + T)

            if metric == "cosine":
                recon_norms = np.linalg.norm(recon, axis=1, keepdims=True).clip(1e-8)
                recon_normed = recon / recon_norms  # (T, D)
                # Higher similarity = better, so rank by descending similarity
                sims = recon_normed @ corpus_normed.T  # (T, N)
                # Rank: count how many corpus items have strictly higher similarity
                true_sims = sims[np.arange(T), true_indices]  # (T,)
                ranks = np.sum(sims > true_sims[:, None], axis=1)  # (T,)
            else:  # l2
                # (T, N) pairwise L2 distances
                dists = np.linalg.norm(recon[:, None, :] - corpus[None, :, :], axis=2)
                true_dists = dists[np.arange(T), true_indices]  # (T,)
                ranks = np.sum(dists < true_dists[:, None], axis=1)  # (T,)

            ranks_all.append(ranks.astype(np.int64))
            lengths_all.append(T)
            trace_idx += 1

    return ranks_all, lengths_all


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    ranks_all: list[np.ndarray],
    lengths_all: list[int],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Compute global Recall@k and MRR from per-step ranks."""
    all_ranks = np.concatenate(ranks_all)
    n = len(all_ranks)

    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean(all_ranks < k))
    metrics["mrr"] = float(np.mean(1.0 / (all_ranks + 1)))
    metrics["median_rank"] = int(np.median(all_ranks))
    metrics["mean_rank"] = float(np.mean(all_ranks))
    metrics["n_segments"] = n
    metrics["n_traces"] = len(lengths_all)
    return metrics


def aggregate_ranks_by_timestep(
    ranks_all: list[np.ndarray],
    lengths_all: list[int],
    ks: tuple[int, ...] = (1, 5, 10),
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Aggregate recall and MRR by timestep position.

    Returns:
        positions: (T_max,)
        recall_by_k: dict mapping f"recall@{k}" -> (T_max,) array of means
        mrr: (T_max,) array
        counts: (T_max,) number of traces contributing to each position
    """
    T_max = max(lengths_all)
    buckets = [[] for _ in range(T_max)]
    for ranks, T in zip(ranks_all, lengths_all):
        for t in range(T):
            buckets[t].append(ranks[t])

    positions = np.arange(T_max)
    counts = np.array([len(b) for b in buckets])

    recall_by_k = {}
    for k in ks:
        recall_by_k[f"recall@{k}"] = np.array(
            [np.mean(np.array(b) < k) if b else 0.0 for b in buckets]
        )

    mrr = np.array(
        [np.mean(1.0 / (np.array(b) + 1)) if b else 0.0 for b in buckets]
    )

    return positions, recall_by_k, mrr, counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_retrieval_vs_timestep(
    positions: np.ndarray,
    recall_by_k: dict[str, np.ndarray],
    mrr: np.ndarray,
    counts: np.ndarray,
    save_path: str | None = None,
    min_samples: int = 5,
):
    valid = counts >= min_samples
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Recall@k vs timestep
    ax = axes[0]
    colors = {"recall@1": "steelblue", "recall@5": "coral", "recall@10": "seagreen"}
    for label, vals in recall_by_k.items():
        color = colors.get(label, None)
        ax.plot(positions[valid], vals[valid], label=label, linewidth=1.5, color=color)
    ax.set_xlabel("Timestep (segment position)")
    ax.set_ylabel("Recall")
    ax.set_title("NN Retrieval Recall vs. timestep")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MRR vs timestep
    ax = axes[1]
    ax.plot(positions[valid], mrr[valid], color="orchid", linewidth=1.5)
    ax.set_xlabel("Timestep (segment position)")
    ax.set_ylabel("MRR")
    ax.set_title("Mean Reciprocal Rank vs. timestep")
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
    parser = argparse.ArgumentParser(description="Nearest-neighbor retrieval analysis")
    parser.add_argument("--model_dir", required=True, help="Directory containing model.eqx and config.json")
    parser.add_argument("--dataset_name", required=True, help="HuggingFace dataset name")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--eval_sampled", action="store_true", help="Use sampled z0 instead of deterministic mu")
    parser.add_argument("--max_traces", type=int, default=500, help="Max traces to evaluate")
    parser.add_argument("--filter_topic", type=str, default="")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine",
                        help="Distance metric for NN retrieval")
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
    config.filter_topic = args.filter_topic

    # Dataset
    dataset = build_dataset(config, embedder, max_traces=args.max_traces)

    if normalizer is not None:
        print("Applying normalizer to embeddings...")
        for i in range(len(dataset.embeddings)):
            dataset.embeddings[i] = normalizer.normalize(dataset.embeddings[i]).astype(np.float32)

    # Build corpus: flatten all embeddings into (N_total, D)
    corpus = build_corpus(dataset)
    print(f"Corpus size: {corpus.shape[0]} segments from {len(dataset)} traces")

    # Compute cumulative offsets for mapping traces back to corpus indices
    trace_lengths = [dataset[i].shape[0] for i in range(len(dataset))]
    corpus_offsets = np.cumsum([0] + trace_lengths[:-1])

    loader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=False, collate_fn=collate_fn,
    )

    # Compute ranks
    mode = "sampled z0~q" if args.eval_sampled else "deterministic z0=mu"
    print(f"Computing NN retrieval ranks ({mode}, metric={args.metric})...")
    rng_key = jax.random.PRNGKey(config.seed) if args.eval_sampled else None
    ranks_all, lengths_all = compute_retrieval_ranks(
        model, loader, corpus, corpus_offsets,
        sampled=args.eval_sampled, rng_key=rng_key, metric=args.metric,
    )

    # Global metrics
    metrics = compute_metrics(ranks_all, lengths_all)
    print(f"\n--- Global NN retrieval ({metrics['n_segments']} segments, "
          f"{metrics['n_traces']} traces) ---")
    for k in (1, 5, 10):
        print(f"  Recall@{k:2d} : {metrics[f'recall@{k}']:.4f}")
    print(f"  MRR       : {metrics['mrr']:.4f}")
    print(f"  Mean rank : {metrics['mean_rank']:.1f}")
    print(f"  Median rank: {metrics['median_rank']}")

    # Per-timestep breakdown
    ks = (1, 5, 10)
    positions, recall_by_k, mrr, counts = aggregate_ranks_by_timestep(ranks_all, lengths_all, ks)

    print(f"\n--- Per-timestep breakdown (first 20) ---")
    header = f"  {'t':>4s}  {'R@1':>8s}  {'R@5':>8s}  {'R@10':>8s}  {'MRR':>8s}  {'n':>6s}"
    print(header)
    for t in range(min(20, len(positions))):
        print(f"  {t:4d}  {recall_by_k['recall@1'][t]:8.4f}  "
              f"{recall_by_k['recall@5'][t]:8.4f}  "
              f"{recall_by_k['recall@10'][t]:8.4f}  "
              f"{mrr[t]:8.4f}  {counts[t]:6d}")

    # Save metrics
    metrics_path = os.path.join(args.model_dir, "nn_retrieval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Plot
    save_path = os.path.join(args.model_dir, "nn_retrieval.png")
    plot_retrieval_vs_timestep(positions, recall_by_k, mrr, counts, save_path)


if __name__ == "__main__":
    main()
