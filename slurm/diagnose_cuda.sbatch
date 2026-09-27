#!/bin/bash

#SBATCH --job-name=diagnose_cuda
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=diagnose_cuda_%j.out
#SBATCH --error=diagnose_cuda_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

echo "=== HOST ==="
hostname

echo "=== NVIDIA-SMI ==="
nvidia-smi

echo "=== NVIDIA DEVICES ==="
ls -l /dev/nvidia* || true

echo "=== CUDA ENVIRONMENT ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-UNSET}"
echo "SLURM_GPUS=${SLURM_GPUS:-UNSET}"

echo "=== PYTORCH CUDA DIAGNOSTIC ==="

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

try:
    torch.cuda.init()
    print("torch.cuda.init(): OK")
except Exception as exc:
    print("torch.cuda.init() ERROR:")
    print(repr(exc))

try:
    print("GPU 0:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("get_device_name(0) ERROR:")
    print(repr(exc))
PY

echo "=== PIP CHECK ==="
python -m pip check