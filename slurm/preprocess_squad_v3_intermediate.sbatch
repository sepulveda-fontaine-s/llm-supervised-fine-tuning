#!/bin/bash
#SBATCH --job-name=squad_v3_tokenize
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=squad_v3_tokenize_%j.out
#SBATCH --error=squad_v3_tokenize_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python -u scripts/preprocess_dataset_v3_intermediate.py