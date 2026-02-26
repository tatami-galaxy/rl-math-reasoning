import sys
sys.path.append('../../../')
sys.path.append('../latent_ode/')
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
from torch.utils.data import DataLoader, random_split
from transformers import HfArgumentParser

from embed import SonarEmbedder
from train_latent_ode import collate_fn, build_dataset
from neural_ode import NeuralODE, mse_batch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:

    # data
    dataset_name: str = field(default="Ujan/deepmath_trace_2")
    dataset_split: str = field(default="train")
    device: str = field(default="cuda")
    min_segments: int = field(default=10)
    embed_batch_size: int = field(default=256)
    segment_by: str = field(default="\n\n")

    # model
    d_ode_hidden: int = field(default=512)   # ODE MLP hidden dim
    ode_depth: int = field(default=3)        # ODE MLP depth

    # training
    seed: int = field(default=42)
    test_size: float = field(default=0.1)
    train_batch_size: int = field(default=32)
    n_epochs: int = field(default=10)
    log_steps: int = field(default=10)
    lr: float = field(default=1e-3)
    n_decode_examples: int = field(default=1)  # SONAR decode examples after eval


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def make_train_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def train_step(model, opt_state, padded, mask, ts):
        loss, grads = eqx.filter_value_and_grad(mse_batch)(
            model, padded, mask, ts
        )
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss

    return train_step


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@eqx.filter_jit
def eval_batch(model: NeuralODE, padded: jax.Array, mask: jax.Array, ts: jax.Array) -> jax.Array:
    """Reconstruction MSE on a batch."""
    return mse_batch(model, padded, mask, ts)


def evaluate(model: NeuralODE, loader: DataLoader) -> float:
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
def reconstruct_batch(model: NeuralODE, padded: jax.Array, mask: jax.Array, ts: jax.Array):
    """Reconstruct embeddings through the ODE.

    Returns:
        x_hat: (B, T_max, D) reconstructed embeddings
    """
    def single(x, m):
        x0 = x[0]
        return model.solve(x0, ts)

    return jax.vmap(single)(padded, mask)


def decode_examples(
    model: NeuralODE,
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
    output_dir = root + "/models/neural_ode/" + config.dataset_name.split("/")[-1]
    print(f"Output dir: {output_dir}")

    # Embedder — SONAR only
    embedder = SonarEmbedder(
        device=config.device,
        batch_size=config.embed_batch_size,
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
    d_embed = embedder.dim  # 1024 for SONAR
    key = jax.random.PRNGKey(config.seed)
    key, model_key = jax.random.split(key)
    model = NeuralODE(d_embed, config.d_ode_hidden, config.ode_depth, key=model_key)

    # Optimizer
    optimizer = optax.adam(config.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(optimizer)

    # Training loop
    writer = SummaryWriter(log_dir=output_dir)
    global_step = 1
    print("Starting training...")
    for epoch in range(config.n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for padded, mask, _, ts in loader:
            padded_jax = jnp.array(padded.numpy())
            mask_jax = jnp.array(mask.numpy())
            ts_jax = jnp.array(ts.numpy())

            model, opt_state, loss = train_step(
                model, opt_state, padded_jax, mask_jax, ts_jax
            )
            batch_loss = float(loss)
            epoch_loss += batch_loss
            n_batches += 1

            if global_step % config.log_steps == 0:
                writer.add_scalar("loss/batch", batch_loss, global_step)

            global_step += 1

        epoch_avg_loss = epoch_loss / n_batches
        writer.add_scalar("loss/epoch", epoch_avg_loss, epoch)
        print(f"Epoch {epoch + 1}/{config.n_epochs}  loss={epoch_avg_loss:.4f}")

    # Evaluation on test set
    print("Evaluating on test set...")
    test_mse = evaluate(model, test_loader)
    writer.add_scalar("mse/test", test_mse, config.n_epochs)
    print(f"Test MSE: {test_mse:.6f}")

    # Decode examples with SONAR
    if config.n_decode_examples > 0:
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
