#!/bin/bash
#SBATCH --job-name=inf_net_base
#SBATCH --output=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/inf-net/inf-net-base/inf_net_base_%A_%a.out
#SBATCH --error=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/inf-net/inf-net-base/inf_net_base_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=124G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx4090
#SBATCH --qos=rtx4090-6hours
#SBATCH --array=1-2%10


mkdir -p /scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/inf-net/inf-net-base

cd /scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension

eval "$(conda shell.bash hook)"
conda activate morph-ext


echo "Activated Conda Environment: $(which python)"
python -c "import torch; print('torch version:', torch.__version__)"
python -c "import torchvision; print('torchvision version:', torchvision.__version__)"

unset PYTHONPATH
export PYTHONPATH=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/external/opacus

echo "Activated Conda Environment: $(which python)"
python -c "import opacus; print('opacus path:', opacus.__file__)"
python -c "import torch; print('torch version:', torch.__version__)"
python -c "import torchvision; print('torchvision version:', torchvision.__version__)"

cd Inf-Net
#["base", "automatic", "psac", "nsgd"],
clp="automatic"
seeds=(1 2)
seed=${seeds[$((SLURM_ARRAY_TASK_ID-1))]}

COMMAND="python MyTrain_LungInf_Unified.py \
  --network NestedUNet \
  --epoch 70 \
  --batchsize 4 \
  --enable_privacy \
  --clipping_strategy "$clp" \
  --epsilon 200 \
  --max_grad_norm 1.2 \
  --lr 1e-4 \
  --run ${SLURM_ARRAY_TASK_ID} \
  --seed ${seed} \
  --backbone Res2Net50"

echo "=========================================="
echo "Job Name: inf_net_base"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Seed: ${seed}"
echo "Node: ${SLURM_NODELIST}"
echo "Command: ${COMMAND}"
echo "=========================================="

eval "${COMMAND}"

echo "Task ${SLURM_ARRAY_TASK_ID} finished."