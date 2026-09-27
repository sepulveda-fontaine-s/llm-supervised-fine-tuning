#!/bin/bash

#SBATCH --job-name=install_optuna
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=install_optuna_%j.out
#SBATCH --error=install_optuna_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

echo "=== INSTALL PROJECT DEPENDENCIES ==="
python -m pip install -e .

echo "=== OPTUNA VERSION ==="
python -c "import optuna; print(optuna.__version__)"

echo "=== PIP CHECK ==="
python -m pip check

echo "=== OPTUNA INSTALL OK ==="