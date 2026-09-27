#!/bin/bash

#SBATCH --job-name=qa_validation_full
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=evaluate_qa_validation_full_%j.out
#SBATCH --error=evaluate_qa_validation_full_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

python -u scripts/evaluate_qa_generation.py \
    --output results/validation_qa_metrics_full.json