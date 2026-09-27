#!/bin/bash
#SBATCH --job-name=qualitative_test_v3
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=qualitative_test_v3_%j.out
#SBATCH --error=qualitative_test_v3_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python -u scripts/analyze_final_test_predictions.py
