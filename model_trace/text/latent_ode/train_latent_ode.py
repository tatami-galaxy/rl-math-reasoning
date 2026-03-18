"""
Run to cluster all deepmath traces into different difficulty levels 
and upload to HF :
python process_deepmath_traces.py --hf_key 

Train latent ode : 
CUDA_VISIBLE_DEVICES=0 python -m model_trace.text.latent_ode.train_latent_ode \\
    --dataset_name Ujan/deepmath_trace_2 --embed_type sonar
CUDA_VISIBLE_DEVICES=0 python -m model_trace.text.latent_ode.train_latent_ode \\
    --dataset_name Ujan/deepmath_trace_3 --embed_type sonar \\
    --beta 1 --d_decoder_depth 5 --eval_sampled --normalize_embeddings
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

from .embed import Embedder, SentenceTransformerEmbedder, SonarEmbedder
from .latent_ode import LatentODE, elbo_batch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:

    # data
    dataset_name: str = field(default="Ujan/deepmath_trace_2")
    dataset_split: str = field(default="train")
    embed_type: str = field(default="sonar")
    normalize_embeddings: bool = field(default=False)  # scaler on SONAR embeddings
    embed_model: str = field(default="sentence-transformers/all-MiniLM-L6-v2")
    device: str = field(default="cuda")
    min_segments: int = field(default=10)
    embed_batch_size: int = field(default=256)
    segment_by: str = field(default="\n\n")

    # model
    d_proj: int = field(default=128)         # input projection dim
    d_encoder: int = field(default=128)      # encoder hidden dim
    d_z: int = field(default=64)             # latent ODE dim
    d_ode_hidden: int = field(default=128)   # ODE MLP hidden dim
    d_decoder_depth: int = field(default=3)  # decoder MLP depth
    beta: float = field(default=1.0)         # KL weight in ELBO
    output_dir: str = field(default="models/latent_ode")

    # training
    seed: int = field(default=42)
    test_size: float = field(default=0.1) 
    train_batch_size: int = field(default=32)
    max_steps: int = field(default=100)
    log_steps: int = field(default=10)
    lr: float = field(default=1e-3)

    # eval
    n_decode_examples: int = field(default=1)   # SONAR decode examples after eval
    eval_sampled: bool = field(default=False)   # use sampled z0 for test MSE
    sonar_sampler: str = field(default="beam")  # SONAR text decoding: "beam", "top_p", "top_k"
    sonar_top_p: float = field(default=0.99)    # p value for top_p / nucleus sampler
    sonar_top_k: int = field(default=50)        # k value for top_k sampler


# ---------------------------------------------------------------------------
# Embedding normalizer
# ---------------------------------------------------------------------------

class EmbeddingNormalizer:
    """Per-dimension scaler

    Fits median and IQR statistics over training embeddings, then:
      normalize(x)   = (x - median) / IQR   — applied before the model
      denormalize(x) = x * IQR + median     — applied after the model, before SONAR decoding
    """

    def fit(self, embeddings: list[np.ndarray]) -> "EmbeddingNormalizer":
        """Compute per-dimension median and IQR from a list of (T_i, D) arrays."""
        all_vecs = np.concatenate(embeddings, axis=0)  # (N, D)
        self.median = np.median(all_vecs, axis=0)       # (D,)
        q1 = np.percentile(all_vecs, 25, axis=0)
        q3 = np.percentile(all_vecs, 75, axis=0)
        self.scale = np.maximum(q3 - q1, 1e-8)         # (D,) IQR, floored for stability
        return self

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """(x - median) / IQR"""
        return (x - self.median) / self.scale

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """x * IQR + median"""
        return x * self.scale + self.median

    def save(self, path: str) -> None:
        np.savez(path, median=self.median, scale=self.scale)
        print(f"Saved normalizer to {path}")

    @classmethod
    def load(cls, path: str) -> "EmbeddingNormalizer":
        data = np.load(path)
        norm = cls()
        norm.median = data["median"]
        norm.scale = data["scale"]
        return norm


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
def eval_batch_sampled(
    model: LatentODE, padded: jax.Array, mask: jax.Array, ts: jax.Array, keys: jax.Array
) -> jax.Array:
    """Reconstruction MSE on a batch using sampled z0 (with reparameterization)."""
    def single(x, m, key):
        z0, _, _ = model.encode(x, m, key)
        zs = model.solve(z0, ts)
        x_hat = model.decode(zs)
        return jnp.sum(m[:, None] * (x_hat - x) ** 2) / jnp.sum(m)

    return jnp.mean(jax.vmap(single)(padded, mask, keys))


def evaluate_sampled(model: LatentODE, loader: DataLoader, rng_key: jax.Array) -> float:
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


@eqx.filter_jit
def reconstruct_batch_sampled(
    model: LatentODE, padded: jax.Array, mask: jax.Array, ts: jax.Array, keys: jax.Array
):
    """Reconstruct embeddings through the ODE (sampled z0).

    Returns:
        x_hat: (B, T_max, D) reconstructed embeddings
    """
    def single(x, m, key):
        z0, _, _ = model.encode(x, m, key)
        zs = model.solve(z0, ts)
        return model.decode(zs)

    return jax.vmap(single)(padded, mask, keys)


def build_sonar_sampler(config: Config):
    """Build a fairseq2 sampler from config, or None for beam search."""
    if config.sonar_sampler == "beam":
        return None
    elif config.sonar_sampler == "top_p":
        from fairseq2.generation import TopPSampler
        return TopPSampler(config.sonar_top_p)
    elif config.sonar_sampler == "top_k":
        from fairseq2.generation import TopKSampler
        return TopKSampler(config.sonar_top_k)
    else:
        raise ValueError(f"Unknown sonar_sampler: {config.sonar_sampler!r}. Use 'beam', 'top_p', or 'top_k'.")


def decode_examples(
    model: LatentODE,
    embedder: SonarEmbedder,
    loader: DataLoader,
    n_examples: int,
    sampled: bool = False,
    rng_key: jax.Array | None = None,
    sonar_sampler=None,
    normalizer: EmbeddingNormalizer | None = None,
):
    """Reconstruct test trajectories through the ODE and decode with SONAR.

    For each example trace, prints the SONAR-decoded text of the original
    embeddings side-by-side with the ODE-reconstructed embeddings.
    """
    mode = "sampled z0~q" if sampled else "deterministic z0=mu"
    collected = 0
    print(f"\n{'=' * 70}")
    print(f"SONAR DECODE EXAMPLES ({mode})")
    print("=" * 70)

    for padded, mask, lengths, ts in loader:
        padded_jax = jnp.array(padded.numpy())
        mask_jax = jnp.array(mask.numpy())
        ts_jax = jnp.array(ts.numpy())

        if sampled:
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, padded.shape[0])
            x_hat = reconstruct_batch_sampled(model, padded_jax, mask_jax, ts_jax, keys)
        else:
            x_hat = reconstruct_batch(model, padded_jax, mask_jax, ts_jax)

        x_hat_np = np.array(x_hat)
        padded_np = padded.numpy()
        lengths_np = lengths.numpy()

        for i in range(len(padded_np)):
            if collected >= n_examples:
                return
            T = int(lengths_np[i])
            orig_emb = padded_np[i, :T]    # (T, D) — normalized
            recon_emb = x_hat_np[i, :T]    # (T, D) — normalized

            # Denormalize before SONAR decoding (LCM §2.3.1 PostNet)
            if normalizer is not None:
                orig_emb = normalizer.denormalize(orig_emb).astype(np.float32)
                recon_emb = normalizer.denormalize(recon_emb).astype(np.float32)
            else:
                orig_emb = orig_emb.astype(np.float32)
                recon_emb = recon_emb.astype(np.float32)

            orig_text = embedder.decode(orig_emb, sampler=sonar_sampler)
            recon_text = embedder.decode(recon_emb, sampler=sonar_sampler)

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

    # output dir
    config.output_dir = config.output_dir + '/' + config.embed_type + '/'
    config.output_dir = config.output_dir + config.dataset_name.split("/")[-1]

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

    # Optionally fit normalizer on training embeddings
    normalizer = None
    if config.normalize_embeddings:
        print("Fitting embedding normalizer on training data...")
        train_embs = [dataset.embeddings[i] for i in train_dataset.indices]
        normalizer = EmbeddingNormalizer().fit(train_embs)
        print(f"  median range : [{normalizer.median.min():.4f}, {normalizer.median.max():.4f}]")
        print(f"  IQR range    : [{normalizer.scale.min():.6f}, {normalizer.scale.max():.4f}]")
        for i in range(len(dataset.embeddings)):
            dataset.embeddings[i] = normalizer.normalize(dataset.embeddings[i]).astype(np.float32)
        print("Normalization applied.")
    else:
        print("Embedding normalization disabled.")

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
    writer = SummaryWriter(log_dir=config.output_dir)
    global_step = 0
    done = False
    print(f"Starting training for {config.max_steps} steps...")
    while not done:
        for padded, mask, _, ts in loader:
            if global_step >= config.max_steps:
                done = True
                break

            B = padded.shape[0]
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, B)

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

    # Evaluation on test set
    print("Evaluating on test set...")
    if config.eval_sampled:
        test_mse = evaluate_sampled(model, test_loader, key)
        writer.add_scalar("mse/test", test_mse, config.max_steps)
        print(f"Test MSE (sampled, z0~q): {test_mse:.6f}")
    else:
        test_mse = evaluate(model, test_loader)
        writer.add_scalar("mse/test", test_mse, config.max_steps)
        print(f"Test MSE (deterministic, z0=mu): {test_mse:.6f}")

    # Decode examples with SONAR
    if config.embed_type == "sonar" and config.n_decode_examples > 0:
        sonar_sampler = build_sonar_sampler(config)
        decode_examples(
            model, embedder, test_loader, config.n_decode_examples,
            sampled=config.eval_sampled, rng_key=key,
            sonar_sampler=sonar_sampler,
            normalizer=normalizer,
        )

    writer.close()

    # Save model, normalizer, and config
    os.makedirs(config.output_dir, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(config.output_dir, "model.eqx"), model)
    if normalizer is not None:
        normalizer.save(os.path.join(config.output_dir, "normalizer.npz"))
    config_dict = vars(config)
    config_dict["d_embed"] = d_embed
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
