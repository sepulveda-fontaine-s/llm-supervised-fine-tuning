#!/bin/bash

#SBATCH --job-name=install_project
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=install_project_%j.out
#SBATCH --error=install_project_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

echo "=== PYTHON ==="
python --version

echo "=== INSTALL PROJECT ==="
python -m pip install -e .

echo "=== PIP CHECK ==="
python -m pip check

echo "=== IMPORT CHECK ==="
python - <<'PY'
import transformers
import datasets
import accelerate
import evaluate
import mlflow
import yaml

print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("evaluate:", evaluate.__version__)
print("mlflow:", mlflow.__version__)
print("PyYAML:", yaml.__version__)
PY