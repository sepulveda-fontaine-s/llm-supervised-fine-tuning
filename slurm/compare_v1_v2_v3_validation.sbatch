#!/bin/bash
#SBATCH --job-name=compare_v1_v2_v3
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=compare_v1_v2_v3_%j.out
#SBATCH --error=compare_v1_v2_v3_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python -u scripts/compare_v1_v2_v3_validation.py \
    --output results/validation_v1_v2_v3_comparison.json