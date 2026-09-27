#!/bin/bash

#SBATCH --job-name=check_qa_metrics
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=check_qa_generation_metrics_%j.out
#SBATCH --error=check_qa_generation_metrics_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

python -m py_compile \
    scripts/evaluate_qa_generation.py

python scripts/evaluate_qa_generation.py \
    --limit 64 \
    --output results/validation_qa_metrics_smoke.json