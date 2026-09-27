#!/bin/bash

#SBATCH --job-name=env_check
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=env_check_%j.out
#SBATCH --error=env_check_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

echo "=== HOST ==="
hostname

echo "=== PYTHON ==="
python --version

echo "=== NVIDIA ==="
nvidia-smi

echo "=== IMPORTS / PYTORCH / CUDA ==="
python - <<'PY'
import torch
import transformers
import datasets
import accelerate
import evaluate
import mlflow
import yaml

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("evaluate:", evaluate.__version__)
print("mlflow:", mlflow.__version__)
print("PyYAML:", yaml.__version__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
PY

echo "=== PIP CHECK ==="
python -m pip check

echo "=== ENVIRONMENT OK ==="