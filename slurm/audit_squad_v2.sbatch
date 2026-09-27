#!/bin/bash

#SBATCH --job-name=audit_squad_v2
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=audit_squad_v2_%j.out
#SBATCH --error=audit_squad_v2_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python scripts/audit_squad_v2.py