#!/bin/bash
#SBATCH --job-name=inf_net_base
#SBATCH --output=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/inf-net/inf-net-base/inf_net_base_%A_%a.out
#SBATCH --error=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/inf-net/inf-net-base/inf_net_base_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx4090
#SBATCH --qos=rtx4090-6hours
#SBATCH --array=1-2%10

# Create logs directory if it doesn't exist
mkdir -p /scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/inf-net/inf-net-base

# Navigate to project directory
cd /scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension

# Activate conda
eval "$(conda shell.bash hook)"
conda activate morph-ext

# Navigate to inf-net code directory
cd Inf-Net
# remove --enable_morphology \
         #  --morph_operation close \ when you don't want morphology

COMMAND="python MyTrain_LungInf_Unified.py \
  --network Inf_Net \
  --epoch 70 \
  --batchsize 24 \
  --lr 1e-4 \
  --run ${SLURM_ARRAY_TASK_ID} \
  --enable_morphology \
  --morph_operation both \
  --backbone Res2Net50"

echo "=========================================="
echo "Job Name: inf_net_base"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Command: ${COMMAND}"
echo "=========================================="

eval "${COMMAND}"

echo "Task ${SLURM_ARRAY_TASK_ID} completed at $(date)"

