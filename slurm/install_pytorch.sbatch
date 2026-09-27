#!/bin/bash

#SBATCH --job-name=install_pytorch
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=install_pytorch_%j.out
#SBATCH --error=install_pytorch_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

echo "=== INSTALL PYTORCH CUDA 12.8 BUILD ==="
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "=== PYTORCH / CUDA CHECK ==="
python -c "import torch; \
print('torch:', torch.__version__); \
print('torch CUDA:', torch.version.cuda); \
print('CUDA available:', torch.cuda.is_available()); \
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "=== PIP CHECK ==="
python -m pip check