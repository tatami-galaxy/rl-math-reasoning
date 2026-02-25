"""
sonar_explore.py — Explore SONAR encode/decode fidelity on DeepMath traces.

Loads a processed DeepMath trace dataset, segments traces by '\n\n', encodes
and decodes with SONAR, then reports per-trace reconstruction failure rates,
segment length statistics, and example segments for success and failure cases.
"""

import random

import numpy as np
import torch
from datasets import load_dataset
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)
from sentence_transformers import SentenceTransformer

# --- Config ---
DATASET_NAME = "Ujan/deepmath_trace_7"
DATASET_SPLIT = "train"
NUM_TRACES = 20       # number of traces to analyse
BATCH_SIZE = 64       # SONAR batch size for encode/decode
SEG_DELIMITER = "\n\n"
NUM_EXAMPLES = 5       # example segments to print per category
MAX_SEQ_LEN = 512
SEED = 42
DTYPE = torch.float16

# Comparison mode: "exact" or "semantic"
COMPARE_MODE = "semantic"
# Semantic similarity threshold (segments below this are "failures")
SIM_THRESHOLD = 0.8
# Sentence-transformer model for semantic comparison
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def segment_trace(trace: str) -> list[str]:
    """Split by double-newline; strip and drop empty segments."""
    return [s.strip() for s in trace.split(SEG_DELIMITER) if s.strip()]


def encode_batched(segs: list[str], model) -> torch.Tensor:
    parts = []
    for i in range(0, len(segs), BATCH_SIZE):
        batch = segs[i : i + BATCH_SIZE]
        parts.append(model.predict(batch, source_lang="eng_Latn"))
        print(f"  encoded {min(i + BATCH_SIZE, len(segs))}/{len(segs)}", end="\r")
    print()
    return torch.cat(parts, dim=0)


def decode_batched(embeddings: torch.Tensor, model) -> list[str]:
    decoded = []
    for i in range(0, len(embeddings), BATCH_SIZE):
        batch = embeddings[i : i + BATCH_SIZE]
        decoded.extend(model.predict(batch, target_lang="eng_Latn", max_seq_len=MAX_SEQ_LEN))
        print(f"  decoded {min(i + BATCH_SIZE, len(embeddings))}/{len(embeddings)}", end="\r")
    print()
    return decoded


# ------------------------------------------------------------------ #
# Comparison functions
# ------------------------------------------------------------------ #

def compare_exact(flat_segs: list[str], flat_decoded: list[str]) -> tuple[list[bool], list[float]]:
    """Exact string match after stripping whitespace.
    Returns (success_mask, scores) where scores are 1.0 or 0.0."""
    success = [o.strip() == d.strip() for o, d in zip(flat_segs, flat_decoded)]
    scores = [1.0 if s else 0.0 for s in success]
    return success, scores


def compare_semantic(flat_segs: list[str], flat_decoded: list[str]) -> tuple[list[bool], list[float]]:
    """Cosine similarity via sentence-transformers.
    Returns (success_mask, scores) where success = sim >= SIM_THRESHOLD."""

    print(f"\nLoading sentence-transformer ({ST_MODEL})...")
    st = SentenceTransformer(ST_MODEL, device=str(DEVICE))

    print("Computing embeddings for original segments...")
    emb_orig = st.encode(flat_segs, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=True)
    print("Computing embeddings for decoded segments...")
    emb_deco = st.encode(flat_decoded, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=True)

    # Row-wise cosine similarity
    dot = np.sum(emb_orig * emb_deco, axis=1)
    norm_orig = np.linalg.norm(emb_orig, axis=1)
    norm_deco = np.linalg.norm(emb_deco, axis=1)
    scores = (dot / (norm_orig * norm_deco + 1e-8)).tolist()

    success = [s >= SIM_THRESHOLD for s in scores]
    return success, scores


COMPARE_FN = {
    "exact": compare_exact,
    "semantic": compare_semantic,
}


def main():
    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    print(f"Loading {DATASET_NAME} ({DATASET_SPLIT})...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    traces = ds["trace"][:NUM_TRACES]
    print(f"  Loaded {len(traces)} traces.")

    # ------------------------------------------------------------------ #
    # Segment
    # ------------------------------------------------------------------ #
    segmented: list[list[str]] = [segment_trace(t) for t in traces]
    seg_counts = [len(s) for s in segmented]
    print(f"  Segments per trace — min: {min(seg_counts)}, "
          f"max: {max(seg_counts)}, mean: {np.mean(seg_counts):.1f}")

    # Flat list + per-trace offsets into it
    flat_segs: list[str] = [seg for segs in segmented for seg in segs]
    offsets = np.cumsum([0] + [len(s) for s in segmented])   # shape: (N+1,)
    print(f"  Total segments: {len(flat_segs)}")

    # ------------------------------------------------------------------ #
    # Load SONAR
    # ------------------------------------------------------------------ #
    print("\nLoading SONAR models...")
    t2vec = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=DEVICE,
        dtype=DTYPE,
    )
    vec2t = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=DEVICE,
        dtype=DTYPE,
    )

    # ------------------------------------------------------------------ #
    # Encode → decode
    # ------------------------------------------------------------------ #
    print("\nEncoding segments...")
    embeddings = encode_batched(flat_segs, t2vec)

    print("Decoding embeddings...")
    flat_decoded = decode_batched(embeddings, vec2t)

    # ------------------------------------------------------------------ #
    # Compare original vs decoded
    # ------------------------------------------------------------------ #
    compare_fn = COMPARE_FN[COMPARE_MODE]
    print(f"\nComparing with mode: {COMPARE_MODE}")
    success, scores = compare_fn(flat_segs, flat_decoded)

    # Per-trace failure fraction
    per_trace_fail: list[float] = []
    for i in range(len(segmented)):
        n = offsets[i + 1] - offsets[i]
        if n == 0:
            continue
        n_fail = sum(1 for m in success[offsets[i] : offsets[i + 1]] if not m)
        per_trace_fail.append(n_fail / n)

    # ------------------------------------------------------------------ #
    # Summary statistics
    # ------------------------------------------------------------------ #
    n_total = len(flat_segs)
    n_success = sum(success)
    n_fail = n_total - n_success

    print("\n" + "=" * 60)
    print(f"RECONSTRUCTION STATISTICS  (mode: {COMPARE_MODE})")
    print("=" * 60)
    print(f"Total segments : {n_total}")
    print(f"Successes      : {n_success:>6d}  ({100 * n_success / n_total:.1f}%)")
    print(f"Failures       : {n_fail:>6d}  ({100 * n_fail / n_total:.1f}%)")

    if COMPARE_MODE == "semantic":
        print(f"\nSimilarity threshold : {SIM_THRESHOLD}")
        print(f"Similarity scores — mean: {np.mean(scores):.3f}, "
              f"median: {np.median(scores):.3f}, "
              f"min: {np.min(scores):.3f}, max: {np.max(scores):.3f}")

    print("\nPer-trace failure fraction:")
    print(f"  mean   = {np.mean(per_trace_fail):.3f}")
    print(f"  median = {np.median(per_trace_fail):.3f}")
    print(f"  min    = {np.min(per_trace_fail):.3f}")
    print(f"  max    = {np.max(per_trace_fail):.3f}")

    success_lens = [len(flat_segs[i]) for i, m in enumerate(success) if m]
    fail_lens    = [len(flat_segs[i]) for i, m in enumerate(success) if not m]

    print("\nAverage segment length (characters):")
    if success_lens:
        print(f"  Successes : n={len(success_lens):>6d}, "
              f"mean={np.mean(success_lens):.1f}, median={np.median(success_lens):.1f}")
    else:
        print("  Successes : none")
    if fail_lens:
        print(f"  Failures  : n={len(fail_lens):>6d}, "
              f"mean={np.mean(fail_lens):.1f}, median={np.median(fail_lens):.1f}")
    else:
        print("  Failures  : none")

    # ------------------------------------------------------------------ #
    # Example segments
    # ------------------------------------------------------------------ #
    success_idx = [i for i, m in enumerate(success) if m]
    fail_idx    = [i for i, m in enumerate(success) if not m]

    def show_examples(indices, label, n=NUM_EXAMPLES):
        sample = random.sample(indices, min(n, len(indices)))
        print(f"\n{'=' * 60}")
        print(f"EXAMPLE {label} SEGMENTS  (original → decoded, first 300 chars)")
        print("=" * 60)
        for idx in sample:
            orig = flat_segs[idx]
            deco = flat_decoded[idx]
            score_str = f"  [score={scores[idx]:.3f}]" if COMPARE_MODE == "semantic" else ""
            print(f"\n[orig]  {repr(orig[:300])}")
            print(f"[deco]  {repr(deco[:300])}{score_str}")

    show_examples(success_idx, "SUCCESS")
    show_examples(fail_idx,    "FAILURE")


if __name__ == "__main__":
    main()
