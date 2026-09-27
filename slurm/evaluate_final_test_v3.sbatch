#!/bin/bash
#SBATCH --job-name=final_test_v3
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=final_test_v3_%j.out
#SBATCH --error=final_test_v3_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python -u scripts/evaluate_final_test_v3.py \
    --output results/test_final_v3.json
