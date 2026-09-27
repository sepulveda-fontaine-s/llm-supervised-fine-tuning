#!/bin/bash

#SBATCH --job-name=report_optuna_squad
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=report_optuna_squad_%j.out
#SBATCH --error=report_optuna_squad_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python scripts/report_optuna_squad.py