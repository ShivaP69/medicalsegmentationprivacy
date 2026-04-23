#!/bin/bash
#SBATCH --job-name=loss-infnet
#SBATCH --time=01:00:00
#SBATCH --qos=rtx4090-6hours
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --output=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/inf-net/loss/%x_%j.out
#SBATCH --error=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/inf-net/loss/%x_%j.err

set -euo pipefail


set -euo pipefail

echo "========================================"
echo "Starting Global Loss Attack (Lung)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate morph-ext   # <-- change if needed

# Go to project directory

PROJECT_ROOT=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension
BASE=$PROJECT_ROOT/Inf-Net/Attack
LOGDIR=$PROJECT_ROOT/logs/inf-net/loss


mkdir -p "$LOGDIR"
cd "$BASE"
eval "$(conda shell.bash hook)"
conda activate morph-ext


echo "Activated Conda Environment: $(which python)"
python -c "import torch; print('torch version:', torch.__version__)"
python -c "import torchvision; print('torchvision version:', torchvision.__version__)"

unset PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/external/opacus

python -c "import opacus; print('opacus path:', opacus.__file__)"
python -c "import torch; print('torch version:', torch.__version__)"
python -c "import torchvision; print('torchvision version:', torchvision.__version__)"
echo "Running attack..."

python global_loss_attack_lung.py \
  --network NestedUNet \
  --epoch 70 \
  --batchsize 4 \
  --run 1 \
  --seed 0 \
  --enable_privacy \
  --epsilon 200 \
  --max_grad_norm 1.2 \
  --clipping_strategy automatic \
  --enable_morphology \
  --morph_operation both \
  --morph_kernel_size 3

echo "========================================"
echo "Finished!"
echo "========================================"