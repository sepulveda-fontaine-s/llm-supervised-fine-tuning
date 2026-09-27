#!/bin/bash

#SBATCH --job-name=check_train_squad
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=check_train_full_squad_%j.out
#SBATCH --error=check_train_full_squad_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

echo "=== PYTHON SYNTAX ==="

python -m py_compile \
    scripts/train_full.py

echo
echo "=== TRAINING CONFIG ==="

python - <<'PY'
import math
import yaml
from datasets import load_from_disk

with open(
    "configs/training.yaml",
    "r",
    encoding="utf-8",
) as f:
    config = yaml.safe_load(f)

training = config["training"]

dataset = load_from_disk(
    "data/tokenized/squad_v2_qwen2_5_0_5b"
)

batch_size = (
    training["per_device_batch_size"]
)

accumulation = (
    training[
        "gradient_accumulation_steps"
    ]
)

micro_batches = math.ceil(
    len(dataset["train"])
    / batch_size
)

steps_per_epoch = math.ceil(
    micro_batches
    / accumulation
)

total_steps = (
    steps_per_epoch
    * training["epochs"]
)

warmup_steps = int(
    total_steps
    * training["warmup_ratio"]
)

print("train examples:", len(dataset["train"]))
print("validation examples:", len(dataset["validation"]))
print("epochs:", training["epochs"])
print("steps per epoch:", steps_per_epoch)
print("total steps:", total_steps)
print("learning rate:", training["learning_rate"])
print("weight decay:", training["weight_decay"])
print("scheduler:", training["scheduler"])
print("warmup ratio:", training["warmup_ratio"])
print("warmup steps:", warmup_steps)

assert len(dataset["train"]) == 29998
assert len(dataset["validation"]) == 5933

assert training["epochs"] == 3

assert training["learning_rate"] == (
    2.5789950779174585e-05
)

assert training["weight_decay"] == (
    0.02433937943676302
)

assert training["scheduler"] == "linear"

assert training["warmup_ratio"] == (
    0.018118910790805944
)

assert steps_per_epoch == 1875
assert total_steps == 5625

print()
print("=== TRAINING CONFIG OK ===")
PY

echo
echo "=== OLD DATASET CHECK ==="

if grep -n "no_robots" \
    scripts/train_full.py \
    configs/training.yaml
then
    echo "ERROR: old no_robots reference found."
    exit 1
else
    echo "No no_robots references found."
fi

echo
echo "=== FULL TRAINING CHECK OK ==="