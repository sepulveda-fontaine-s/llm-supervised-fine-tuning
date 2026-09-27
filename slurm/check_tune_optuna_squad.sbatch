#!/bin/bash

#SBATCH --job-name=check_optuna_squad
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=check_tune_optuna_squad_%j.out
#SBATCH --error=check_tune_optuna_squad_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

echo "=== PYTHON SYNTAX ==="
python -m py_compile scripts/tune_optuna.py

echo
echo "=== CONFIG CHECK ==="

python - <<'PY'
import math
import yaml
from datasets import load_from_disk

CONFIG_PATH = "configs/training.yaml"
DATASET_PATH = "data/tokenized/squad_v2_qwen2_5_0_5b"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

training = config["training"]
tuning = config["tuning"]
generation = config["generation"]

dataset = load_from_disk(DATASET_PATH)

batch_size = training["per_device_batch_size"]
accumulation_steps = training["gradient_accumulation_steps"]

micro_batches_per_epoch = math.ceil(
    len(dataset["train"]) / batch_size
)

optimizer_steps_per_epoch = math.ceil(
    micro_batches_per_epoch / accumulation_steps
)

full_training_steps = (
    optimizer_steps_per_epoch
    * training["epochs"]
)

print("train examples:", len(dataset["train"]))
print("validation examples:", len(dataset["validation"]))
print("test examples:", len(dataset["test"]))

print("batch size:", batch_size)
print("gradient accumulation:", accumulation_steps)
print("effective batch size:", batch_size * accumulation_steps)

print("optimizer steps per epoch:", optimizer_steps_per_epoch)
print("training epochs:", training["epochs"])
print("full training steps:", full_training_steps)

print("Optuna trials:", tuning["n_trials"])
print("Optuna max steps/trial:", tuning["max_optimizer_steps"])
print("validation subset:", tuning["validation_examples"])

print(
    "learning rate range:",
    tuning["learning_rate_min"],
    tuning["learning_rate_max"],
)

print(
    "weight decay range:",
    tuning["weight_decay_min"],
    tuning["weight_decay_max"],
)

print(
    "warmup ratio range:",
    tuning["warmup_ratio_min"],
    tuning["warmup_ratio_max"],
)

print("schedulers:", tuning["schedulers"])
print("evaluation interval:", tuning["evaluation_interval"])
print("pruning start:", tuning["pruning_start_step"])
print("study:", tuning["study_name"])
print("storage:", tuning["storage"])

print("generation:", generation)

assert len(dataset["train"]) == 29998
assert len(dataset["validation"]) == 5933
assert len(dataset["test"]) == 5940

assert optimizer_steps_per_epoch == 1875
assert full_training_steps == 5625

assert training["epochs"] == 3

assert tuning["n_trials"] == 12
assert tuning["max_optimizer_steps"] == 600
assert tuning["validation_examples"] == 128

assert tuning["learning_rate_min"] == 2.0e-5
assert tuning["learning_rate_max"] == 8.0e-5

assert tuning["evaluation_interval"] == 100
assert tuning["pruning_start_step"] == 300

assert tuning["study_name"] == (
    "qwen_squad_v2_full_sft_hparam_search_v1"
)

assert tuning["storage"] == (
    "sqlite:///results/optuna_squad_v2_v1.db"
)

assert generation["do_sample"] is False
assert generation["repetition_penalty"] == 1.0
assert generation["max_new_tokens"] == 64

print()
print("=== CONFIG OK ===")
PY

echo
echo "=== OLD DATASET REFERENCES ==="

if grep -n "no_robots" \
    scripts/tune_optuna.py \
    configs/training.yaml
then
    echo "ERROR: old no_robots reference found."
    exit 1
else
    echo "No no_robots references found."
fi

echo
echo "=== OPTUNA SQUAD CHECK OK ==="