#!/bin/bash

#SBATCH --job-name=train_full_sft
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=train_full_%j.out
#SBATCH --error=train_full_%j.err
#SBATCH --mail-type=END,FAIL

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export MLFLOW_ALLOW_FILE_STORE=true

python scripts/train_full.py