#!/bin/bash
#SBATCH --job-name=loss_attack
#SBATCH --time=00:20:00
#SBATCH --qos=rtx4090-6hours
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --output=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/loss/Duke/%x_%j.out
#SBATCH --error=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/logs/loss/Duke/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension
BASE=$PROJECT_ROOT/Attack
LOGDIR=$PROJECT_ROOT/logs/loss/Duke
mkdir -p "$LOGDIR"

module purge

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

PYFILE=$BASE/global_loss_attack.py

DATASET="UMN"
MODEL_NAME="unet"
N_CLASSES=2
DPSGD=false
SEED=6 # 5 and 6
#"None" "flat", "automatic", "psac", "normalized_sgd"
CLIPPING="None"
MORPHOLOGY=true
OPERATION="close"
KERNEL_SIZE=3
USE_MORPH=true
CONDITIONAL_POINT="None"
RETINAL_LAYER_WISE=false
EXCL_3_4=False
POLICY_TYPE="None"
DEEP_SUPERVISION=false
EPSILON=200
BATCH_SIZE=8
NUM_ITERATIONS=100
IMAGE_SIZE=224
G_RATIO=0.5
LEARNABLE_RADIUS=false
IMAGE_DIR=""

CMD=(
  python "$PYFILE"
  --dataset "$DATASET"
  --batch_size "$BATCH_SIZE"
  --num_iterations "$NUM_ITERATIONS"
  --excluding_3_4_layer "$EXCL_3_4"
  --n_classes "$N_CLASSES"
  --model_name "$MODEL_NAME"
  --image_size "$IMAGE_SIZE"
  --g_ratio "$G_RATIO"
  --morphology "$MORPHOLOGY"
  --operation "$OPERATION"
  --kernel_size "$KERNEL_SIZE"
  --learnable_radius "$LEARNABLE_RADIUS"
  --use_morph "$USE_MORPH"
  --conditional_point "$CONDITIONAL_POINT"
  --retinal_layer_wise "$RETINAL_LAYER_WISE"
  --policy_type "$POLICY_TYPE"
  --DPSGD "$DPSGD"
  --deep_supervision "$DEEP_SUPERVISION"
  --seed "$SEED"
  --epsilon "$EPSILON"
  --clipping "$CLIPPING"
)

if [[ -n "$IMAGE_DIR" ]]; then
  CMD+=( --image_dir "$IMAGE_DIR" )
fi

echo "Running command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"