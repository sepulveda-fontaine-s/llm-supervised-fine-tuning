#!/bin/bash
#SBATCH --job-name=squad_v3_train
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=squad_v3_train_%j.out
#SBATCH --error=squad_v3_train_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export MLFLOW_ALLOW_FILE_STORE=true

python -u scripts/train_full_v3_intermediate.py
