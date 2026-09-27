#!/bin/bash

#SBATCH --job-name=check_squad_tokenized
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=check_squad_tokenized_%j.out
#SBATCH --error=check_squad_tokenized_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python - <<'PY'
from datasets import load_from_disk

PATH = "data/tokenized/squad_v2_qwen2_5_0_5b"
MAX_LENGTH = 1024

dataset = load_from_disk(PATH)

expected_sizes = {
    "train": 29998,
    "validation": 5933,
    "test": 5940,
}

for split_name in [
    "train",
    "validation",
    "test",
]:
    split = dataset[split_name]

    max_length = 0
    min_supervised = None
    zero_supervised = 0

    for example in split:
        length = len(example["input_ids"])

        supervised = sum(
            label != -100
            for label in example["labels"]
        )

        max_length = max(
            max_length,
            length,
        )

        if min_supervised is None:
            min_supervised = supervised
        else:
            min_supervised = min(
                min_supervised,
                supervised,
            )

        if supervised == 0:
            zero_supervised += 1

    print(f"=== {split_name.upper()} ===")
    print("examples:", len(split))
    print("max length:", max_length)
    print(
        "minimum supervised tokens:",
        min_supervised,
    )
    print(
        "zero-supervised examples:",
        zero_supervised,
    )

    assert len(split) == expected_sizes[split_name]
    assert max_length <= MAX_LENGTH
    assert zero_supervised == 0
    assert min_supervised > 0

print("\n=== TOKENIZED DATASET CHECK OK ===")
PY