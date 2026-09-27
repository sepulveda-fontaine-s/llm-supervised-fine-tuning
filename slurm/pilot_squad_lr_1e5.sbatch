#!/bin/bash

#SBATCH --job-name=pilot_squad_1e5
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=pilot_squad_lr_1e5_%j.out
#SBATCH --error=pilot_squad_lr_1e5_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export MLFLOW_ALLOW_FILE_STORE=true

python scripts/train_pilot.py \
    --learning-rate 1e-5 \
    --weight-decay 0.01