#!/bin/bash

#SBATCH --job-name=check_squad_preproc
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=check_squad_preprocessing_%j.out
#SBATCH --error=check_squad_preprocessing_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python - <<'PY'
import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer

from llm_fine_tuning.preprocessing import (
    build_sft_example,
    build_squad_messages,
)

with open(
    "configs/training.yaml",
    "r",
    encoding="utf-8",
) as f:
    config = yaml.safe_load(f)

dataset = load_from_disk(
    config["data"]["path"]
)

tokenizer = AutoTokenizer.from_pretrained(
    config["model"]["name"]
)


def inspect(example, name):
    messages = build_squad_messages(example)

    encoded = build_sft_example(
        messages,
        tokenizer,
    )

    labels = encoded["labels"]

    supervised_ids = [
        token_id
        for token_id, label in zip(
            encoded["input_ids"],
            labels,
        )
        if label != -100
    ]

    supervised_text = tokenizer.decode(
        supervised_ids,
        skip_special_tokens=False,
    )

    print(f"\n=== {name} ===")
    print(
        "answer:",
        messages[-1]["content"],
    )
    print(
        "total tokens:",
        len(encoded["input_ids"]),
    )
    print(
        "supervised tokens:",
        len(supervised_ids),
    )
    print(
        "supervised text:",
        repr(supervised_text),
    )

    assert len(supervised_ids) > 0


answerable = next(
    example
    for example in dataset["train"]
    if len(example["answers"]["text"]) > 0
)

unanswerable = next(
    example
    for example in dataset["train"]
    if len(example["answers"]["text"]) == 0
)

inspect(
    answerable,
    "ANSWERABLE",
)

inspect(
    unanswerable,
    "UNANSWERABLE",
)

print("\n=== SQUAD PREPROCESSING CHECK OK ===")
PY