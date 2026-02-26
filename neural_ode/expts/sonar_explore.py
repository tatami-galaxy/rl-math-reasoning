"""
sonar_explore.py — Explore SONAR encode/decode fidelity on DeepMath traces.

Loads a processed DeepMath trace dataset, segments traces by '\n\n', encodes
and decodes with SONAR, then reports per-trace reconstruction failure rates,
segment length statistics, and example segments for success and failure cases.
"""

import sys
sys.path.append('../../')
from utils import get_root_dir
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml
from datasets import load_dataset
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)
from sentence_transformers import SentenceTransformer

# --- Config ---
DATASET_NAME = "Ujan/deepmath_trace_5"
DATASET_SPLIT = "train"
NUM_TRACES = 20       # number of traces to analyse
BATCH_SIZE = 64       # SONAR batch size for encode/decode
SEG_DELIMITER = "\n\n"
NUM_EXAMPLES = 5       # example segments to print per category
MAX_SEQ_LEN = 512
SEED = 42
DTYPE = torch.float16

# Comparison mode: "exact", "semantic", or "perplexity"
COMPARE_MODE = "semantic"
# Semantic similarity threshold (segments below this are "failures")
SIM_THRESHOLD = 0.8
PPL_THRESHOLD = 300.0
# Sentence-transformer model for semantic comparison
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Perplexity model and threshold (segments above this are "failures")
PPL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# dir
ROOT = get_root_dir()
REPORT_DIR = ROOT + "/reports/sonar_expts"

# seed, device
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


def compare_perplexity(_flat_segs: list[str], flat_decoded: list[str]) -> tuple[list[bool], list[float]]:
    """Per-segment perplexity of decoded text via a causal LM.
    Returns (success_mask, scores) where scores are perplexity values
    and success = ppl <= PPL_THRESHOLD."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading perplexity model ({PPL_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        PPL_MODEL, torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()

    print("Computing perplexity for decoded segments...")
    ppls: list[float] = []
    for i, text in enumerate(flat_decoded):
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(model.device)
        if input_ids.shape[1] <= 1:
            ppls.append(float("inf"))
            continue
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        ppls.append(torch.exp(outputs.loss).item())
        if (i + 1) % 50 == 0 or i == len(flat_decoded) - 1:
            print(f"  {i + 1}/{len(flat_decoded)}", end="\r")
    print()

    success = [p <= PPL_THRESHOLD for p in ppls]
    return success, ppls


COMPARE_FN = {
    "exact": compare_exact,
    "semantic": compare_semantic,
    "perplexity": compare_perplexity,
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
    elif COMPARE_MODE == "perplexity":
        finite_scores = [s for s in scores if np.isfinite(s)]
        print(f"\nPerplexity threshold : {PPL_THRESHOLD}")
        if finite_scores:
            print(f"Perplexity scores — mean: {np.mean(finite_scores):.1f}, "
                  f"median: {np.median(finite_scores):.1f}, "
                  f"min: {np.min(finite_scores):.1f}, max: {np.max(finite_scores):.1f}")
        n_inf = len(scores) - len(finite_scores)
        if n_inf:
            print(f"  ({n_inf} segments too short to score)")

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
            if COMPARE_MODE == "semantic":
                score_str = f"  [sim={scores[idx]:.3f}]"
            elif COMPARE_MODE == "perplexity":
                score_str = f"  [ppl={scores[idx]:.1f}]"
            else:
                score_str = ""
            print(f"\n[orig]  {repr(orig[:300])}")
            print(f"[deco]  {repr(deco[:300])}{score_str}")

    show_examples(success_idx, "SUCCESS")
    show_examples(fail_idx,    "FAILURE")

    # ------------------------------------------------------------------ #
    # Save report
    # ------------------------------------------------------------------ #
    save_report(
        flat_segs, flat_decoded, scores, success,
        per_trace_fail, success_lens, fail_lens,
        success_idx, fail_idx,
    )


def save_report(
    flat_segs, flat_decoded, scores, success,
    per_trace_fail, success_lens, fail_lens,
    success_idx, fail_idx,
):
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Dataset tag for filename, e.g. "deepmath_trace_7"
    ds_tag = DATASET_NAME.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ds_tag}_{COMPARE_MODE}_{timestamp}.yaml"
    path = os.path.join(REPORT_DIR, filename)

    n_total = len(flat_segs)
    n_success = sum(success)
    n_fail = n_total - n_success

    report: dict = {
        "config": {
            "dataset": DATASET_NAME,
            "split": DATASET_SPLIT,
            "num_traces": NUM_TRACES,
            "seg_delimiter": repr(SEG_DELIMITER),
            "compare_mode": COMPARE_MODE,
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "seed": SEED,
        },
        "summary": {
            "total_segments": n_total,
            "successes": n_success,
            "success_pct": round(100 * n_success / n_total, 1),
            "failures": n_fail,
            "failure_pct": round(100 * n_fail / n_total, 1),
        },
        "per_trace_failure_fraction": {
            "mean": round(float(np.mean(per_trace_fail)), 3),
            "median": round(float(np.median(per_trace_fail)), 3),
            "min": round(float(np.min(per_trace_fail)), 3),
            "max": round(float(np.max(per_trace_fail)), 3),
        },
        "segment_length_chars": {},
    }

    if success_lens:
        report["segment_length_chars"]["successes"] = {
            "n": len(success_lens),
            "mean": round(float(np.mean(success_lens)), 1),
            "median": round(float(np.median(success_lens)), 1),
        }
    if fail_lens:
        report["segment_length_chars"]["failures"] = {
            "n": len(fail_lens),
            "mean": round(float(np.mean(fail_lens)), 1),
            "median": round(float(np.median(fail_lens)), 1),
        }

    # Mode-specific fields
    if COMPARE_MODE == "semantic":
        report["config"]["sim_threshold"] = SIM_THRESHOLD
        report["config"]["st_model"] = ST_MODEL
        report["scores"] = {
            "mean": round(float(np.mean(scores)), 3),
            "median": round(float(np.median(scores)), 3),
            "min": round(float(np.min(scores)), 3),
            "max": round(float(np.max(scores)), 3),
        }
    elif COMPARE_MODE == "perplexity":
        report["config"]["ppl_threshold"] = PPL_THRESHOLD
        report["config"]["ppl_model"] = PPL_MODEL
        finite = [s for s in scores if np.isfinite(s)]
        if finite:
            report["scores"] = {
                "mean": round(float(np.mean(finite)), 1),
                "median": round(float(np.median(finite)), 1),
                "min": round(float(np.min(finite)), 1),
                "max": round(float(np.max(finite)), 1),
                "n_inf": len(scores) - len(finite),
            }

    # Example segments
    def sample_examples(indices, n=NUM_EXAMPLES):
        sample = random.sample(indices, min(n, len(indices)))
        examples = []
        for idx in sample:
            entry = {
                "original": flat_segs[idx][:500],
                "decoded": flat_decoded[idx][:500],
            }
            if COMPARE_MODE == "semantic":
                entry["similarity"] = round(scores[idx], 3)
            elif COMPARE_MODE == "perplexity":
                entry["perplexity"] = round(scores[idx], 1) if np.isfinite(scores[idx]) else "inf"
            examples.append(entry)
        return examples

    report["examples"] = {
        "successes": sample_examples(success_idx),
        "failures": sample_examples(fail_idx),
    }

    with open(path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)

    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()
