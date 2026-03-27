"""
Vector field property analysis for a trained Latent ODE model.

Computes properties of the learned vector field f(z) along reconstructed
latent trajectories: speed, divergence, curvature, eigenvalue spectrum, etc.

Built incrementally — each property is self-contained and can be run
independently via --properties flag.

Usage:
    python -m model_trace.text.latent_ode.analysis.field_properties \
        --model_dir models/latent_ode/qwen/deepmath_trace_3 \
        --dataset_name Ujan/deepmath_trace_3 \
        --filter_topic Algebra \
        --properties speed
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
from ..embed import SentenceTransformerEmbedder, SonarEmbedder, QwenEmbedder
from .recon_quality import load_model


# ---------------------------------------------------------------------------
# Latent trajectory extraction
# ---------------------------------------------------------------------------

@eqx.filter_jit
def _extract_latent_trajectory_deterministic(
    model: LatentODE, x: jax.Array, mask: jax.Array, ts: jax.Array,
) -> jax.Array:
    """Encode (deterministic mu) and integrate ODE, returning latent trajectory.

    Returns:
        zs: (T, d_z) latent states at each timestep
    """
    mu, _ = model.encoder(x, mask)
    return model.solve(mu, ts)


@eqx.filter_jit
def _extract_latent_trajectory_sampled(
    model: LatentODE, x: jax.Array, mask: jax.Array, ts: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Encode (sampled z0) and integrate ODE, returning latent trajectory."""
    z0, _, _ = model.encode(x, mask, key)
    return model.solve(z0, ts)


def extract_latent_trajectories(
    model: LatentODE,
    loader: DataLoader,
    sampled: bool = False,
    rng_key: jax.Array | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract latent trajectories z(t) for all traces in the loader.

    Returns:
        zs_all: list of (T_i, d_z) arrays
        lengths: list of ints
    """
    zs_all = []
    lengths_all = []

    for padded, mask, lengths, ts in loader:
        padded_jax = jnp.array(padded.numpy())
        mask_jax = jnp.array(mask.numpy())
        ts_jax = jnp.array(ts.numpy())
        lengths_np = lengths.numpy()

        for i in range(len(lengths_np)):
            T = int(lengths_np[i])
            x_i = padded_jax[i]
            m_i = mask_jax[i]

            if sampled:
                rng_key, subkey = jax.random.split(rng_key)
                zs = _extract_latent_trajectory_sampled(model, x_i, m_i, ts_jax, subkey)
            else:
                zs = _extract_latent_trajectory_deterministic(model, x_i, m_i, ts_jax)

            zs_all.append(np.array(zs[:T]))
            lengths_all.append(T)

    return zs_all, lengths_all


# ---------------------------------------------------------------------------
# 1. Speed: ||f(z)||
# ---------------------------------------------------------------------------

def compute_speed(
    model: LatentODE, zs_all: list[np.ndarray],
) -> list[np.ndarray]:
    """Compute ||f(z(t))|| at each timestep for each trajectory.

    Returns:
        speeds: list of (T_i,) arrays
    """
    # Unwrap the raw ODE function for evaluation outside diffrax
    ode_fn = model.ode_func

    @eqx.filter_jit
    def _speed_single(z: jax.Array) -> jax.Array:
        """Compute ||f(z)|| for a single latent state."""
        fz = ode_fn(0.0, z, None)
        return jnp.linalg.norm(fz)

    _speed_batch = jax.jit(jax.vmap(_speed_single))

    speeds = []
    for zs in zs_all:
        s = _speed_batch(jnp.array(zs))
        speeds.append(np.array(s))

    return speeds


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_by_timestep(
    per_step_values: list[np.ndarray],
    lengths: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate variable-length per-step values by timestep position.

    Returns:
        positions: (T_max,)
        means: (T_max,)
        stds: (T_max,)
        counts: (T_max,)
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


def aggregate_by_normalized_position(
    per_step_values: list[np.ndarray],
    lengths: list[int],
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate values by normalized position t/T in [0, 1].

    This allows comparison across traces of different lengths.

    Returns:
        bin_centers: (n_bins,)
        means: (n_bins,)
        stds: (n_bins,)
        counts: (n_bins,)
    """
    buckets = [[] for _ in range(n_bins)]
    for vals, T in zip(per_step_values, lengths):
        for t in range(T):
            frac = t / T
            b = min(int(frac * n_bins), n_bins - 1)
            buckets[b].append(vals[t])

    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    means = np.array([np.mean(b) if b else np.nan for b in buckets])
    stds = np.array([np.std(b) if b else np.nan for b in buckets])
    counts = np.array([len(b) for b in buckets])

    return bin_centers, means, stds, counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_speed(
    positions: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    counts: np.ndarray,
    norm_centers: np.ndarray,
    norm_means: np.ndarray,
    norm_stds: np.ndarray,
    norm_counts: np.ndarray,
    save_path: str | None = None,
    min_samples: int = 5,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Speed vs absolute timestep
    ax = axes[0]
    valid = counts >= min_samples
    ax.plot(positions[valid], means[valid], color="steelblue", linewidth=1.5)
    ax.fill_between(
        positions[valid],
        (means - stds)[valid],
        (means + stds)[valid],
        alpha=0.2, color="steelblue",
    )
    ax.set_xlabel("Timestep (segment position)")
    ax.set_ylabel("||f(z)||")
    ax.set_title("Vector field speed vs. timestep")
    ax.grid(True, alpha=0.3)

    # Speed vs normalized position
    ax = axes[1]
    valid = norm_counts >= min_samples
    ax.plot(norm_centers[valid], norm_means[valid], color="coral", linewidth=1.5)
    ax.fill_between(
        norm_centers[valid],
        (norm_means - norm_stds)[valid],
        (norm_means + norm_stds)[valid],
        alpha=0.2, color="coral",
    )
    ax.set_xlabel("Normalized position (t / T)")
    ax.set_ylabel("||f(z)||")
    ax.set_title("Vector field speed vs. normalized position")
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
    parser = argparse.ArgumentParser(description="Vector field property analysis")
    parser.add_argument("--model_dir", required=True,
                        help="Directory containing model.eqx and config.json")
    parser.add_argument("--dataset_name", required=True,
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--eval_sampled", action="store_true",
                        help="Use sampled z0 instead of deterministic mu")
    parser.add_argument("--max_traces", type=int, default=500)
    parser.add_argument("--filter_topic", type=str, default="")
    parser.add_argument("--embed_batch_size", type=int, default=128)
    parser.add_argument("--properties", nargs="+", default=["speed"],
                        choices=["speed"],
                        help="Which field properties to compute")
    args = parser.parse_args()

    # Load config (no JAX model yet — keep GPU free for embedder)
    print(f"Loading config from {args.model_dir}")
    with open(os.path.join(args.model_dir, "config.json")) as f:
        config_dict = json.load(f)
    d_embed = config_dict.pop("d_embed")
    config = Config(**config_dict)

    config.dataset_name = args.dataset_name
    config.dataset_split = args.dataset_split
    config.filter_topic = args.filter_topic
    config.embed_batch_size = args.embed_batch_size

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
    elif config.embed_type == "qwen":
        embedder = QwenEmbedder(
            model_name=config.embed_model,
            device=config.device,
            batch_size=config.embed_batch_size,
            instruction=config.embed_instruction,
        )
    else:
        raise NotImplementedError(f"Embedding type: {config.embed_type}")

    dataset = build_dataset(config, embedder)

    del embedder
    import torch
    torch.cuda.empty_cache()

    # Load model + normalizer
    print(f"Loading model from {args.model_dir}")
    skeleton = LatentODE(d_embed, config, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(
        os.path.join(args.model_dir, "model.eqx"), skeleton
    )

    normalizer_path = os.path.join(args.model_dir, "normalizer.npz")
    normalizer = EmbeddingNormalizer.load(normalizer_path) if os.path.exists(normalizer_path) else None
    if normalizer is not None:
        print("Loaded normalizer from", normalizer_path)
        for i in range(len(dataset.embeddings)):
            dataset.embeddings[i] = normalizer.normalize(dataset.embeddings[i]).astype(np.float32)

    if args.max_traces < len(dataset):
        indices = np.random.default_rng(0).choice(len(dataset), args.max_traces, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    loader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=False, collate_fn=collate_fn,
    )

    # Extract latent trajectories
    mode = "sampled z0~q" if args.eval_sampled else "deterministic z0=mu"
    print(f"Extracting latent trajectories ({mode})...")
    rng_key = jax.random.PRNGKey(config.seed) if args.eval_sampled else None
    zs_all, lengths = extract_latent_trajectories(
        model, loader, sampled=args.eval_sampled, rng_key=rng_key,
    )
    print(f"  {len(zs_all)} trajectories, d_z={zs_all[0].shape[1]}")

    # --- Speed ---
    if "speed" in args.properties:
        print("Computing speed ||f(z)||...")
        speeds = compute_speed(model, zs_all)

        # Summary
        all_speeds = np.concatenate(speeds)
        print(f"\n--- Speed summary ({len(lengths)} traces) ---")
        print(f"  mean={all_speeds.mean():.6f}  std={all_speeds.std():.6f}  "
              f"median={np.median(all_speeds):.6f}")
        print(f"  min={all_speeds.min():.6f}  max={all_speeds.max():.6f}")

        # Aggregate by absolute timestep
        pos, means, stds, counts = aggregate_by_timestep(speeds, lengths)

        print(f"\n--- Speed vs timestep (first 20) ---")
        print(f"  {'t':>4s}  {'mean':>10s}  {'std':>10s}  {'n':>6s}")
        for t in range(min(20, len(pos))):
            if counts[t] >= 2:
                print(f"  {t:4d}  {means[t]:10.6f}  {stds[t]:10.6f}  {counts[t]:6d}")

        # Aggregate by normalized position
        norm_centers, norm_means, norm_stds, norm_counts = aggregate_by_normalized_position(
            speeds, lengths,
        )

        print(f"\n--- Speed vs normalized position ---")
        print(f"  {'pos':>6s}  {'mean':>10s}  {'std':>10s}  {'n':>6s}")
        for b in range(len(norm_centers)):
            if norm_counts[b] >= 2:
                print(f"  {norm_centers[b]:6.2f}  {norm_means[b]:10.6f}  "
                      f"{norm_stds[b]:10.6f}  {norm_counts[b]:6d}")

        # Plot
        save_path = os.path.join(args.model_dir, "field_speed.png")
        plot_speed(pos, means, stds, counts,
                   norm_centers, norm_means, norm_stds, norm_counts,
                   save_path)

        # Save metrics
        metrics = {
            "speed_mean": float(all_speeds.mean()),
            "speed_std": float(all_speeds.std()),
            "speed_median": float(np.median(all_speeds)),
            "speed_min": float(all_speeds.min()),
            "speed_max": float(all_speeds.max()),
            "n_traces": len(lengths),
        }
        metrics_path = os.path.join(args.model_dir, "field_speed_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
