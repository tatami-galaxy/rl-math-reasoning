import sys
sys.path.append('../../../')
sys.path.append('../latent_ode/')

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

from neural_ode import NeuralODE
from train_neural_ode import Config, collate_fn
from train_latent_ode import build_dataset
from embed import SonarEmbedder


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(model_dir: str) -> tuple[NeuralODE, Config, int]:
    with open(os.path.join(model_dir, "config.json")) as f:
        config_dict = json.load(f)

    d_embed = config_dict.pop("d_embed")
    config = Config(**config_dict)

    skeleton = NeuralODE(d_embed, config.d_ode_hidden, config.ode_depth, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(
        os.path.join(model_dir, "model.eqx"), skeleton
    )
    return model, config, d_embed


# ---------------------------------------------------------------------------
# Collect x₀ (first embedding of each trace)
# ---------------------------------------------------------------------------

def collect_x0(loader: DataLoader) -> np.ndarray:
    """Collect the first valid embedding from each trace in the dataset."""
    x0_list = []
    for padded, mask, _, _ts in loader:
        # padded: (B, T_max, D), first timestep is always valid
        x0_list.append(padded[:, 0, :].numpy())
    return np.concatenate(x0_list, axis=0)  # (N, d_embed)


# ---------------------------------------------------------------------------
# Phase portrait: PCA projection + quiver plot
# ---------------------------------------------------------------------------

def phase_portrait(
    model: NeuralODE,
    x0_samples: np.ndarray,
    grid_resolution: int = 20,
    save_path: str | None = None,
):
    """
    1. PCA on x₀ samples → 2D
    2. Build a 2D grid spanning the x₀ cloud
    3. Map grid points back to d_embed via PCA inverse transform
    4. Evaluate f_θ at each grid point
    5. Project velocity vectors to 2D (tangent projection)
    6. Plot scatter of x₀ + quiver of f_θ
    """
    # 1. PCA
    pca = PCA(n_components=2)
    x0_2d = pca.fit_transform(x0_samples)  # (N, 2)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    # 2. Grid in PCA space
    pad = 0.1
    x_range = x0_2d[:, 0].max() - x0_2d[:, 0].min()
    y_range = x0_2d[:, 1].max() - x0_2d[:, 1].min()
    x_lin = np.linspace(x0_2d[:, 0].min() - pad * x_range,
                        x0_2d[:, 0].max() + pad * x_range, grid_resolution)
    y_lin = np.linspace(x0_2d[:, 1].min() - pad * y_range,
                        x0_2d[:, 1].max() + pad * y_range, grid_resolution)
    xx, yy = np.meshgrid(x_lin, y_lin)
    grid_2d = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (G, 2)

    # 3. Inverse PCA: 2D grid → d_embed
    grid_embed = pca.inverse_transform(grid_2d)  # (G, d_embed)

    # 4. Evaluate vector field f_θ at each grid point
    @eqx.filter_jit
    def eval_field(xs):
        return jax.vmap(lambda x: model.ode_func(0.0, x, None))(xs)

    velocities_embed = np.array(eval_field(jnp.array(grid_embed, dtype=jnp.float32)))  # (G, d_embed)

    # 5. Project velocity vectors to 2D
    velocities_2d = velocities_embed @ pca.components_.T  # (G, 2)

    # Normalise arrow lengths for visual clarity
    norms = np.linalg.norm(velocities_2d, axis=1, keepdims=True).clip(1e-8)
    velocities_2d_norm = velocities_2d / norms

    # 6. Plot
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(x0_2d[:, 0], x0_2d[:, 1], s=6, alpha=0.35, color="steelblue", label="$x_0$ samples")
    ax.quiver(
        grid_2d[:, 0], grid_2d[:, 1],
        velocities_2d_norm[:, 0], velocities_2d_norm[:, 1],
        np.linalg.norm(velocities_2d, axis=1),
        cmap="plasma", alpha=0.85,
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Neural ODE — vector field $f_\\theta(x)$ in SONAR space (PCA projection)")
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

    # Build embedder + dataset
    embedder = SonarEmbedder(
        device=config.device,
        batch_size=config.embed_batch_size,
    )
    dataset = build_dataset(config, embedder)

    # Optionally subsample for speed
    if args.max_traces < len(dataset):
        indices = np.random.default_rng(0).choice(len(dataset), args.max_traces, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False, collate_fn=collate_fn)

    # Collect x₀
    print("Collecting x₀ samples...")
    x0_samples = collect_x0(loader)
    print(f"x₀ shape: {x0_samples.shape}")

    # Phase portrait
    save_path = os.path.join(args.model_dir, "phase_portrait.png")
    phase_portrait(model, x0_samples, grid_resolution=args.grid_resolution, save_path=save_path)


if __name__ == "__main__":
    main()
