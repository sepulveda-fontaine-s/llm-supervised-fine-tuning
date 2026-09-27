#!/bin/bash
#SBATCH --job-name=squad_v2_v2_data
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

mkdir -p logs results

python -u scripts/create_squad_v2_balanced_v2.py