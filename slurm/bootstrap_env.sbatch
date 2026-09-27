#!/bin/bash

#SBATCH --job-name=bootstrap_env
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=bootstrap_env_%j.out
#SBATCH --error=bootstrap_env_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

echo "=== HOST ==="
hostname

echo "=== NVIDIA ==="
nvidia-smi

echo "=== SYSTEM PYTHON ==="
python3 --version

echo "=== CREATE PROJECT VENV ==="
python3 -m venv --copies .venv

source .venv/bin/activate

echo "=== VENV PYTHON ==="
python --version

echo "=== UPGRADE PACKAGING TOOLS ==="
python -m pip install --upgrade pip setuptools wheel

echo "=== BOOTSTRAP COMPLETE ==="