#!/bin/bash
#SBATCH --job-name=morphology_oct
#SBATCH --time=1-00:00:00
#SBATCH --qos=rtx4090-1day
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --output=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/oct/UMN/%x_%j.out
#SBATCH --error=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/oct/UMN/%x_%j.err
export CUDA_LAUNCH_BLOCKING=1
set -euo pipefail

BASE=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension
LOGDIR=$BASE/logs/oct/UMN
mkdir -p "$LOGDIR"

module purge

cd "$BASE"
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

JOBID="${SLURM_JOB_ID:-manual}"

# ---- EDIT THESE ----
PYFILE=$BASE/dp_extension_codes/train_one_gpu.py
DATASET="UMN"
svgradient=False
ITERS=100
BS=8
OP="close"
KS=3
LEARNR=false
MORPH=true
TEST=true
lr=3e-4
gn=1
deeps=False
#["None", "flat", "automatic", "psac", "normalized_sgd"],
clp='None'
dpsgd=false
# --------------------

#model_names=("LFUNet" "NestedUNet" "unet")
model_names=("unet")
#morph_approaches=("training" "validation" "None")
morph_approaches=("None")
seeds=(5 6)


for model in "${model_names[@]}"; do
  for point in "${morph_approaches[@]}"; do
    for seed in "${seeds[@]}"; do

      RUN_TAG="${model}_DPSGD${dpsgd}_${DATASET}_Morph_op${OP}_k${KS}_learnR${LEARNR}_cond${point}_clip${clp}_seed${seed}"
      OUTRUN="${LOGDIR}/${RUN_TAG}_${JOBID}.log"

      echo "=========================================================="
      echo "RUN: ${RUN_TAG}"
      echo "Logging to: ${OUTRUN}"
      echo "=========================================================="

      if [[ "$point" == "None" ]]; then
        python "$PYFILE" \
          --dataset "$DATASET" \
          --save_gradient "$svgradient" \
          --n_classes 2 \
          --deep_supervision "$deeps" \
          --model_name "$model" \
          --num_iterations "$ITERS" \
          --batch_size "$BS" \
          --DPSGD "$dpsgd" \
          --learning_rate "$lr" \
          --max_grad_norm "$gn" \
          --morphology "$MORPH" \
          --operation "$OP" \
          --kernel_size "$KS" \
          --conditional_morph False \
          --use_morph True \
          --test "$TEST" \
          --retinal_layer_wise False \
          --clipping "$clp" \
          --seed "$seed" \
          2>&1 | tee "$OUTRUN"
      else
        python "$PYFILE" \
          --dataset "$DATASET" \
          --save_gradient "$svgradient" \
          --n_classes 2 \
          --deep_supervision "$deeps" \
          --model_name "$model" \
          --num_iterations "$ITERS" \
          --batch_size "$BS" \
          --DPSGD "$dpsgd" \
          --learning_rate "$lr" \
          --max_grad_norm "$gn" \
          --morphology "$MORPH" \
          --operation "$OP" \
          --kernel_size "$KS" \
          --conditional_morph True \
          --conditional_point "$point" \
          --use_morph False \
          --test "$TEST" \
          --retinal_layer_wise False \
          --clipping "$clp" \
          --seed "$seed" \
          2>&1 | tee "$OUTRUN"
      fi

    done
  done
done

echo "All runs finished."