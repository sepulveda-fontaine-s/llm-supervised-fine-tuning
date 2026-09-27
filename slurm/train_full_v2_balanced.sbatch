#!/bin/bash
#SBATCH --job-name=squad_v2_v2_train
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export MLFLOW_ALLOW_FILE_STORE=true

python -u scripts/train_full_v2_balanced.py