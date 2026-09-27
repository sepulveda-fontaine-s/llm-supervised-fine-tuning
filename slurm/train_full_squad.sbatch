#!/bin/bash

#SBATCH --job-name=train_full_squad
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=train_full_squad_%j.out
#SBATCH --error=train_full_squad_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export MLFLOW_ALLOW_FILE_STORE=true

python -u scripts/train_full.py