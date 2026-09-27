#!/bin/bash

#SBATCH --job-name=prepare_squad_v2
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=prepare_squad_v2_splits_%j.out
#SBATCH --error=prepare_squad_v2_splits_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python scripts/prepare_squad_v2_splits.py