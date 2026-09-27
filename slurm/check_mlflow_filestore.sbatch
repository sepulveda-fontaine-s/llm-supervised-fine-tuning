#!/bin/bash

#SBATCH --job-name=check_mlflow
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=check_mlflow_%j.out
#SBATCH --error=check_mlflow_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export MLFLOW_ALLOW_FILE_STORE=true

python - <<'PY'
import os
from pathlib import Path

print("=== ENV ===")
print("MLFLOW_ALLOW_FILE_STORE:", os.getenv("MLFLOW_ALLOW_FILE_STORE"))

import mlflow

print("\n=== MLFLOW ===")
print("version:", mlflow.__version__)

tracking_dir = Path("mlruns").resolve()
tracking_dir.mkdir(parents=True, exist_ok=True)

tracking_uri = tracking_dir.as_uri()

print("tracking URI:", tracking_uri)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("mlflow_environment_test")

print("=== MLFLOW FILE STORE OK ===")
PY