#!/bin/bash

#SBATCH --job-name=preprocess_squad_v2
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=preprocess_dataset_%j.out
#SBATCH --error=preprocess_dataset_%j.err
#SBATCH --mail-type=END,FAIL

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python scripts/preprocess_dataset.py