"""
python process_deepmath_traces.py --hf_key 
"""

import sys
sys.path.append('../../')

import math
from collections import defaultdict
from dataclasses import dataclass, field

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import HfArgumentParser


@dataclass
class DataPrepArgs:
    seed: int = field(default=42)
    dataset_name: str = field(default="zwhe99/DeepMath-103K")
    dataset_split: str = field(default="train")
    output_dir: str = field(default="deepmath_traces")
    hf_key: str = field(default=None)


def main():
    parser = HfArgumentParser(DataPrepArgs)
    args = parser.parse_args_into_dataclasses()[0]
    if args.hf_key is None:
        raise ValueError("Pass in hf_key")

    api = HfApi(token=args.hf_key)
    hf_username = api.whoami()["name"]

    print(f"Loading dataset: {args.dataset_name} ({args.dataset_split} split)")
    # ['question', 'final_answer', 'difficulty', 'topic', 'r1_solution_1', 'r1_solution_2', 'r1_solution_3']
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    # Merge the three solution columns into a single "trace" column.
    # Each original row becomes 3 rows, with question prepended to the trace.
    def merge_solutions(batch):
        traces, difficulties, final_answers, topics = [], [], [], []
        for question, sol1, sol2, sol3, diff, ans, topic in zip(
            batch["question"],
            batch["r1_solution_1"],
            batch["r1_solution_2"],
            batch["r1_solution_3"],
            batch["difficulty"],
            batch["final_answer"],
            batch["topic"],
        ):
            for sol in (sol1, sol2, sol3):
                traces.append(question + "\n\n" + sol)
                difficulties.append(diff)
                final_answers.append(ans)
                topics.append(topic)
        return {"trace": traces, "difficulty": difficulties,
                "final_answer": final_answers, "topic": topics}

    print("Merging solution columns into 'trace'...")
    dataset = dataset.map(
        merge_solutions,
        batched=True,
        remove_columns=dataset.column_names,
    ).shuffle(seed=args.seed)

    # Group rows by difficulty bin
    clusters: dict[int, list[dict]] = defaultdict(list)

    # difficulty is floored
    print("Clustering by difficulty...")
    for row in dataset:
        difficulty = row["difficulty"]
        cluster_key = math.floor(difficulty)
        clusters[cluster_key].append({
            "trace": row["trace"],
            "difficulty": difficulty,
            "final_answer": row["final_answer"],
            "topic": row["topic"],
        })
    # delete -1 cluster
    if -1 in clusters:
        del clusters[-1]

    # Upload each cluster as a separate HuggingFace dataset
    for cluster_key, rows in sorted(clusters.items()):
        repo_id = f"{hf_username}/deepmath_trace_{cluster_key}"
        hf_dataset = Dataset.from_list(rows)
        print(f"  difficulty {cluster_key} → {len(rows):>6d} rows → uploading to {repo_id}")
        hf_dataset.push_to_hub(repo_id, token=args.hf_key)

    print(f"\nDone. Uploaded {len(clusters)} datasets to HuggingFace.")


if __name__ == "__main__":
    main()
