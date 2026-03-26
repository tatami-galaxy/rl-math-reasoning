"""
Extrapolation analysis for a trained Latent ODE model.

Given a trace with T segments, encode only the first k segments, integrate the
ODE for all T steps, and compare predicted embeddings at positions k..T-1
against the ground-truth embeddings.  This tests whether the vector field has
learned meaningful dynamics beyond autoencoding.

Two baselines are included for comparison:
  - Last-observed repeat: copy embedding[k-1] to all future positions
  - Linear extrapolation: project from the last two observed embeddings

Metrics are reported per observation fraction and per "steps ahead" horizon.

Usage:
    python -m model_trace.text.latent_ode.analysis.extrapolation \
        --model_dir models/latent_ode/qwen/deepmath_trace_3 \
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
from ..embed import SentenceTransformerEmbedder, SonarEmbedder, QwenEmbedder
from .recon_quality import load_model


# ---------------------------------------------------------------------------
# Extrapolation: encode partial, integrate full
# ---------------------------------------------------------------------------

@eqx.filter_jit
def _extrapolate_deterministic(
    model: LatentODE, x: jax.Array, mask_partial: jax.Array, ts: jax.Array,
) -> jax.Array:
    """Encode from partial observation (deterministic mu), integrate for all T steps.

    Args:
        x: (T, D) trajectory embeddings (zeroed beyond observation cutoff)
        mask_partial: (T,) True only for observed positions
        ts: (T,) timestamps

    Returns:
        x_hat: (T, D) decoded trajectory from partial observation
    """
    mu, _ = model.encoder(x, mask_partial)
    zs = model.solve(mu, ts)
    return model.decode(zs)


@eqx.filter_jit
def _extrapolate_sampled(
    model: LatentODE, x: jax.Array, mask_partial: jax.Array, ts: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Encode from partial observation (sampled z0), integrate for all T steps."""
    z0, _, _ = model.encode(x, mask_partial, key)
    zs = model.solve(z0, ts)
    return model.decode(zs)


def _make_partial_inputs(
    x_np: np.ndarray, mask_np: np.ndarray, k: int,
) -> tuple[jax.Array, jax.Array]:
    """Build masked inputs on the Python side so k doesn't enter JIT as a tracer.

    Returns JAX arrays with positions >= k zeroed/masked.
    """
    x_partial = x_np.copy()
    x_partial[k:] = 0.0
    mask_partial = mask_np.copy()
    mask_partial[k:] = False
    return jnp.array(x_partial), jnp.array(mask_partial)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_repeat(x: np.ndarray, k: int) -> np.ndarray:
    """Repeat the last observed embedding for all future positions.

    Args:
        x: (T, D) original embeddings
        k: cutoff — positions 0..k-1 are observed

    Returns:
        pred: (T, D) where pred[t] = x[k-1] for t >= k, x[t] for t < k
    """
    pred = x.copy()
    pred[k:] = x[k - 1]
    return pred


def baseline_linear(x: np.ndarray, k: int) -> np.ndarray:
    """Linear extrapolation from the last two observed embeddings.

    pred[t] = x[k-1] + (t - k + 1) * (x[k-1] - x[k-2])   for t >= k

    Args:
        x: (T, D) original embeddings
        k: cutoff (must be >= 2)

    Returns:
        pred: (T, D)
    """
    pred = x.copy()
    slope = x[k - 1] - x[k - 2]  # (D,)
    T = x.shape[0]
    for t in range(k, T):
        pred[t] = x[k - 1] + (t - k + 1) * slope
    return pred


# ---------------------------------------------------------------------------
# Per-step error computation
# ---------------------------------------------------------------------------

def _step_errors(orig: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-step MSE and cosine similarity between orig and pred.

    Args:
        orig: (T, D)
        pred: (T, D)

    Returns:
        mse: (T,)
        cos: (T,)
    """
    mse = np.mean((orig - pred) ** 2, axis=1)
    dot = np.sum(orig * pred, axis=1)
    norm_o = np.linalg.norm(orig, axis=1).clip(1e-8)
    norm_p = np.linalg.norm(pred, axis=1).clip(1e-8)
    cos = dot / (norm_o * norm_p)
    return mse, cos


# ---------------------------------------------------------------------------
# Collect extrapolation errors across dataset
# ---------------------------------------------------------------------------

def collect_extrapolation_errors(
    model: LatentODE,
    loader: DataLoader,
    fractions: list[float],
    sampled: bool = False,
    rng_key: jax.Array | None = None,
) -> dict:
    """Run extrapolation for each trace at each observation fraction.

    Returns a dict keyed by fraction, each containing:
        "ode_mse", "ode_cos": list of (horizon,) arrays — error in extrapolated region
        "repeat_mse", "repeat_cos": same for repeat baseline
        "linear_mse", "linear_cos": same for linear baseline
        "observed_mse", "observed_cos": list of (k,) arrays — sanity check on observed region
        "horizons": list of ints — number of extrapolated steps per trace
        "ks": list of ints — cutoff per trace
        "Ts": list of ints — total length per trace
    """
    results = {}
    for frac in fractions:
        results[frac] = {
            "ode_mse": [], "ode_cos": [],
            "repeat_mse": [], "repeat_cos": [],
            "linear_mse": [], "linear_cos": [],
            "observed_mse": [], "observed_cos": [],
            "horizons": [], "ks": [], "Ts": [],
        }

    for padded, mask, lengths, ts in loader:
        padded_np = padded.numpy()
        mask_np = mask.numpy()
        lengths_np = lengths.numpy()

        for i in range(len(lengths_np)):
            T = int(lengths_np[i])
            orig = padded_np[i, :T]  # (T, D)

            for frac in fractions:
                k = max(2, int(T * frac))  # need >= 2 for linear baseline
                if k >= T:
                    continue  # nothing to extrapolate

                horizon = T - k

                # ODE extrapolation — build partial inputs outside JIT
                x_partial, mask_partial = _make_partial_inputs(padded_np[i], mask_np[i], k)
                ts_jax = jnp.array(ts.numpy())

                if sampled:
                    rng_key, subkey = jax.random.split(rng_key)
                    x_hat = _extrapolate_sampled(model, x_partial, mask_partial, ts_jax, subkey)
                else:
                    x_hat = _extrapolate_deterministic(model, x_partial, mask_partial, ts_jax)

                x_hat_np = np.array(x_hat[:T])

                # Errors in extrapolated region (k..T-1)
                ode_mse, ode_cos = _step_errors(orig[k:], x_hat_np[k:])
                rep_mse, rep_cos = _step_errors(orig[k:], baseline_repeat(orig, k)[k:])
                lin_mse, lin_cos = _step_errors(orig[k:], baseline_linear(orig, k)[k:])

                # Errors in observed region (0..k-1) — sanity check
                obs_mse, obs_cos = _step_errors(orig[:k], x_hat_np[:k])

                r = results[frac]
                r["ode_mse"].append(ode_mse)
                r["ode_cos"].append(ode_cos)
                r["repeat_mse"].append(rep_mse)
                r["repeat_cos"].append(rep_cos)
                r["linear_mse"].append(lin_mse)
                r["linear_cos"].append(lin_cos)
                r["observed_mse"].append(obs_mse)
                r["observed_cos"].append(obs_cos)
                r["horizons"].append(horizon)
                r["ks"].append(k)
                r["Ts"].append(T)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_steps_ahead(
    values_list: list[np.ndarray],
    max_horizon: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate variable-length horizon arrays by steps-ahead position.

    Returns:
        positions: (H,) steps ahead indices
        means: (H,)
        stds: (H,)
        counts: (H,)
    """
    buckets = [[] for _ in range(max_horizon)]
    for vals in values_list:
        for s in range(min(len(vals), max_horizon)):
            buckets[s].append(vals[s])

    # Trim trailing empty buckets
    while buckets and len(buckets[-1]) == 0:
        buckets.pop()

    H = len(buckets)
    positions = np.arange(H)
    means = np.array([np.mean(b) if b else np.nan for b in buckets])
    stds = np.array([np.std(b) if b else np.nan for b in buckets])
    counts = np.array([len(b) for b in buckets])

    return positions, means, stds, counts


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(results: dict, fractions: list[float]):
    """Print a global summary table for each observation fraction."""
    print(f"\n{'=' * 75}")
    print("EXTRAPOLATION SUMMARY")
    print("=" * 75)

    for frac in fractions:
        r = results[frac]
        n = len(r["horizons"])
        if n == 0:
            print(f"\n  frac={frac:.0%}: no traces (all too short)")
            continue

        ode_mse = np.concatenate(r["ode_mse"]).mean()
        ode_cos = np.concatenate(r["ode_cos"]).mean()
        rep_mse = np.concatenate(r["repeat_mse"]).mean()
        rep_cos = np.concatenate(r["repeat_cos"]).mean()
        lin_mse = np.concatenate(r["linear_mse"]).mean()
        lin_cos = np.concatenate(r["linear_cos"]).mean()
        obs_mse = np.concatenate(r["observed_mse"]).mean()
        obs_cos = np.concatenate(r["observed_cos"]).mean()
        mean_horizon = np.mean(r["horizons"])

        print(f"\n  Observe {frac:.0%} of trace  ({n} traces, mean horizon={mean_horizon:.1f} steps)")
        print(f"    {'':20s}  {'MSE':>10s}  {'Cosine':>10s}")
        print(f"    {'Observed region':20s}  {obs_mse:10.6f}  {obs_cos:10.4f}")
        print(f"    {'ODE extrapolation':20s}  {ode_mse:10.6f}  {ode_cos:10.4f}")
        print(f"    {'Repeat baseline':20s}  {rep_mse:10.6f}  {rep_cos:10.4f}")
        print(f"    {'Linear baseline':20s}  {lin_mse:10.6f}  {lin_cos:10.4f}")


def print_per_step_table(results: dict, frac: float, max_rows: int = 20):
    """Print a per-steps-ahead breakdown for a single fraction."""
    r = results[frac]
    if not r["horizons"]:
        return

    _, ode_means, _, counts = aggregate_by_steps_ahead(r["ode_mse"])
    _, rep_means, _, _ = aggregate_by_steps_ahead(r["repeat_mse"])
    _, lin_means, _, _ = aggregate_by_steps_ahead(r["linear_mse"])
    _, ode_cos_means, _, _ = aggregate_by_steps_ahead(r["ode_cos"])

    n = min(max_rows, len(ode_means))
    print(f"\n--- Per-steps-ahead breakdown (frac={frac:.0%}, first {n}) ---")
    print(f"  {'ahead':>5s}  {'ODE_MSE':>10s}  {'Rep_MSE':>10s}  {'Lin_MSE':>10s}  {'ODE_Cos':>10s}  {'n':>6s}")
    for s in range(n):
        if counts[s] < 2:
            continue
        print(
            f"  {s:5d}  {ode_means[s]:10.6f}  {rep_means[s]:10.6f}  "
            f"{lin_means[s]:10.6f}  {ode_cos_means[s]:10.4f}  {counts[s]:6d}"
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_extrapolation(
    results: dict,
    fractions: list[float],
    save_dir: str | None = None,
    min_samples: int = 5,
):
    """Plot MSE and cosine vs steps-ahead for each observation fraction."""
    n_frac = len(fractions)
    fig, axes = plt.subplots(n_frac, 2, figsize=(14, 5 * n_frac), squeeze=False)

    for row, frac in enumerate(fractions):
        r = results[frac]
        if not r["horizons"]:
            for col in range(2):
                axes[row, col].set_title(f"frac={frac:.0%} — no data")
            continue

        pos_ode, ode_mse_m, ode_mse_s, counts = aggregate_by_steps_ahead(r["ode_mse"])
        _, rep_mse_m, _, _ = aggregate_by_steps_ahead(r["repeat_mse"])
        _, lin_mse_m, _, _ = aggregate_by_steps_ahead(r["linear_mse"])
        _, ode_cos_m, ode_cos_s, _ = aggregate_by_steps_ahead(r["ode_cos"])
        _, rep_cos_m, _, _ = aggregate_by_steps_ahead(r["repeat_cos"])
        _, lin_cos_m, _, _ = aggregate_by_steps_ahead(r["linear_cos"])

        valid = counts >= min_samples

        # MSE plot
        ax = axes[row, 0]
        ax.plot(pos_ode[valid], ode_mse_m[valid], label="ODE", color="steelblue", linewidth=1.5)
        ax.fill_between(
            pos_ode[valid],
            (ode_mse_m - ode_mse_s)[valid],
            (ode_mse_m + ode_mse_s)[valid],
            alpha=0.15, color="steelblue",
        )
        ax.plot(pos_ode[valid], rep_mse_m[valid], label="Repeat last", color="gray",
                linewidth=1.2, linestyle="--")
        ax.plot(pos_ode[valid], lin_mse_m[valid], label="Linear extrap.", color="coral",
                linewidth=1.2, linestyle="--")
        ax.set_xlabel("Steps ahead")
        ax.set_ylabel("MSE")
        ax.set_title(f"Extrapolation MSE — observe {frac:.0%}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cosine plot
        ax = axes[row, 1]
        ax.plot(pos_ode[valid], ode_cos_m[valid], label="ODE", color="steelblue", linewidth=1.5)
        ax.fill_between(
            pos_ode[valid],
            (ode_cos_m - ode_cos_s)[valid],
            (ode_cos_m + ode_cos_s)[valid],
            alpha=0.15, color="steelblue",
        )
        ax.plot(pos_ode[valid], rep_cos_m[valid], label="Repeat last", color="gray",
                linewidth=1.2, linestyle="--")
        ax.plot(pos_ode[valid], lin_cos_m[valid], label="Linear extrap.", color="coral",
                linewidth=1.2, linestyle="--")
        ax.set_xlabel("Steps ahead")
        ax.set_ylabel("Cosine similarity")
        ax.set_title(f"Extrapolation cosine — observe {frac:.0%}")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir is not None:
        save_path = os.path.join(save_dir, "extrapolation.png")
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extrapolation analysis for Latent ODE")
    parser.add_argument("--model_dir", required=True,
                        help="Directory containing model.eqx and config.json")
    parser.add_argument("--dataset_name", required=True,
                        help="HuggingFace dataset name (e.g. Ujan/deepmath_trace_3)")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--eval_sampled", action="store_true",
                        help="Use sampled z0 instead of deterministic mu")
    parser.add_argument("--max_traces", type=int, default=500,
                        help="Max traces to evaluate")
    parser.add_argument("--filter_topic", type=str, default="")
    parser.add_argument("--embed_batch_size", type=int, default=128)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                        help="Observation fractions to test (default: 0.25 0.5 0.75)")
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

    # Free embedder GPU memory before loading JAX model
    del embedder
    import torch
    torch.cuda.empty_cache()

    # Load JAX model + normalizer
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

    # Run extrapolation
    mode = "sampled z0~q" if args.eval_sampled else "deterministic z0=mu"
    print(f"Computing extrapolation errors ({mode})...")
    print(f"  Observation fractions: {args.fractions}")
    rng_key = jax.random.PRNGKey(config.seed) if args.eval_sampled else None

    results = collect_extrapolation_errors(
        model, loader, args.fractions,
        sampled=args.eval_sampled, rng_key=rng_key,
    )

    # Print results
    print_summary(results, args.fractions)
    for frac in args.fractions:
        print_per_step_table(results, frac)

    # Save metrics
    metrics = {}
    for frac in args.fractions:
        r = results[frac]
        if not r["horizons"]:
            continue
        metrics[f"frac_{frac}"] = {
            "n_traces": len(r["horizons"]),
            "mean_horizon": float(np.mean(r["horizons"])),
            "ode_mse": float(np.concatenate(r["ode_mse"]).mean()),
            "ode_cos": float(np.concatenate(r["ode_cos"]).mean()),
            "repeat_mse": float(np.concatenate(r["repeat_mse"]).mean()),
            "repeat_cos": float(np.concatenate(r["repeat_cos"]).mean()),
            "linear_mse": float(np.concatenate(r["linear_mse"]).mean()),
            "linear_cos": float(np.concatenate(r["linear_cos"]).mean()),
            "observed_mse": float(np.concatenate(r["observed_mse"]).mean()),
            "observed_cos": float(np.concatenate(r["observed_cos"]).mean()),
        }

    metrics_path = os.path.join(args.model_dir, "extrapolation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Plot
    plot_extrapolation(results, args.fractions, save_dir=args.model_dir)


if __name__ == "__main__":
    main()
