import sys
sys.path.append('../../../')

import os
import json
import argparse

import numpy as np
import torch
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from latent_ode import LatentODE
from train_latent_ode import Config, build_dataset, collate_fn
from embed import SentenceTransformerEmbedder


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(model_dir: str) -> tuple[LatentODE, Config, int]:
    with open(os.path.join(model_dir, "config.json")) as f:
        config_dict = json.load(f)

    d_embed = config_dict.pop("d_embed")
    config = Config(**config_dict)

    # Build skeleton with correct shapes, then overwrite weights from disk
    skeleton = LatentODE(d_embed, config, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(
        os.path.join(model_dir, "model.eqx"), skeleton
    )
    return model, config, d_embed


# ---------------------------------------------------------------------------
# Collect z₀ = mu values from a dataset
# ---------------------------------------------------------------------------

def collect_z0(model: LatentODE, loader: DataLoader) -> np.ndarray:
    """Run the encoder over all batches and collect mu (z₀) values."""
    @eqx.filter_jit
    def encode_batch(padded, mask):
        def single(x, m):
            mu, _ = model.encoder(x, m)
            return mu
        return jax.vmap(single)(padded, mask)  # (B, d_z)

    z0_list = []
    for padded, mask, _, _ts in loader:
        mu_batch = encode_batch(
            jnp.array(padded.numpy()),
            jnp.array(mask.numpy()),
        )
        z0_list.append(np.array(mu_batch))

    return np.concatenate(z0_list, axis=0)  # (N, d_z)


# ---------------------------------------------------------------------------
# Phase portrait: PCA projection + quiver plot
# ---------------------------------------------------------------------------

def phase_portrait(
    model: LatentODE,
    z0_samples: np.ndarray,
    grid_resolution: int = 20,
    save_path: str | None = None,
):
    """
    1. PCA on z₀ samples → 2D
    2. Build a 2D grid spanning the z₀ cloud
    3. Map grid points back to d_z via PCA inverse transform
    4. Evaluate f_θ at each grid point
    5. Project velocity vectors to 2D (tangent projection, no mean subtraction)
    6. Plot scatter of z₀ + quiver of f_θ
    """
    # 1. PCA
    pca = PCA(n_components=2)
    z0_2d = pca.fit_transform(z0_samples)  # (N, 2)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    # 2. Grid in PCA space
    pad = 0.1  # fractional padding around the data cloud
    x_range = z0_2d[:, 0].max() - z0_2d[:, 0].min()
    y_range = z0_2d[:, 1].max() - z0_2d[:, 1].min()
    x_lin = np.linspace(z0_2d[:, 0].min() - pad * x_range,
                        z0_2d[:, 0].max() + pad * x_range, grid_resolution)
    y_lin = np.linspace(z0_2d[:, 1].min() - pad * y_range,
                        z0_2d[:, 1].max() + pad * y_range, grid_resolution)
    xx, yy = np.meshgrid(x_lin, y_lin)
    grid_2d = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (G, 2)

    # 3. Inverse PCA: 2D grid → d_z
    grid_dz = pca.inverse_transform(grid_2d)  # (G, d_z)

    # 4. Evaluate vector field f_θ at each grid point
    @eqx.filter_jit
    def eval_field(zs):
        return jax.vmap(lambda z: model.ode_func(0.0, z, None))(zs)

    velocities_dz = np.array(eval_field(jnp.array(grid_dz, dtype=jnp.float32)))  # (G, d_z)

    # 5. Project velocity vectors to 2D.
    # Velocities are tangent vectors: project with components_ only (no mean shift).
    # pca.components_: (2, d_z), so vel_2d = vel_dz @ components_.T → (G, 2)
    velocities_2d = velocities_dz @ pca.components_.T  # (G, 2)

    # Normalise arrow lengths for visual clarity (direction only)
    norms = np.linalg.norm(velocities_2d, axis=1, keepdims=True).clip(1e-8)
    velocities_2d_norm = velocities_2d / norms

    # 6. Plot
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(z0_2d[:, 0], z0_2d[:, 1], s=6, alpha=0.35, color="steelblue", label="$z_0$ samples")
    ax.quiver(
        grid_2d[:, 0], grid_2d[:, 1],
        velocities_2d_norm[:, 0], velocities_2d_norm[:, 1],
        np.linalg.norm(velocities_2d, axis=1),   # colour by unnormalised magnitude
        cmap="plasma", alpha=0.85,
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Latent ODE — vector field $f_\\theta(z)$ (PCA projection)")
    ax.legend(loc="upper right", markerscale=2)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Directory containing model.eqx and config.json")
    parser.add_argument("--grid_resolution", type=int, default=20, help="Grid points per axis")
    parser.add_argument("--max_traces", type=int, default=500, help="Max traces to use for PCA")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_dir}")
    model, config, d_embed = load_model(args.model_dir)

    # Build embedder + dataset (same config as training)
    embedder = SentenceTransformerEmbedder(
        model_name=config.embed_model,
        device=config.device,
        batch_size=config.embed_batch_size,
    )
    # can have test samples too
    # this should be fine if model does not overfit
    # check test MSE first
    dataset = build_dataset(config, embedder)

    # Optionally subsample for speed
    if args.max_traces < len(dataset):
        indices = np.random.default_rng(0).choice(len(dataset), args.max_traces, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False, collate_fn=collate_fn)

    # Collect z₀
    print("Collecting z₀ samples...")
    z0_samples = collect_z0(model, loader)
    print(f"z₀ shape: {z0_samples.shape}")

    # Phase portrait
    save_path = os.path.join(args.model_dir, "phase_portrait.png")
    phase_portrait(model, z0_samples, grid_resolution=args.grid_resolution, save_path=save_path)


if __name__ == "__main__":
    main()
