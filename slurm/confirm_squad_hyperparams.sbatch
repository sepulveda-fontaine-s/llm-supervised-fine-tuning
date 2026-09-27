#!/bin/bash

#SBATCH --job-name=confirm_squad_hparams
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=confirm_squad_hyperparams_%j.out
#SBATCH --error=confirm_squad_hyperparams_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

python scripts/confirm_squad_hyperparams.py