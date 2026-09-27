#!/bin/bash

#SBATCH --job-name=inspect_qa_failures
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=inspect_unanswerable_failures_%j.out
#SBATCH --error=inspect_unanswerable_failures_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python - <<'PY'
import json

PATH = "results/validation_qa_metrics_smoke.json"

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["fine_tuned"]["predictions"]

failures = [
    item
    for item in results
    if (
        not item["answerable"]
        and not item["abstained"]
    )
]

print(
    "unanswerable non-abstentions:",
    len(failures),
)
print()

for i, item in enumerate(
    failures,
    start=1,
):
    print("=" * 80)
    print("FAILURE", i)
    print(
        "prediction:",
        repr(item["prediction"]),
    )
    print()
PY