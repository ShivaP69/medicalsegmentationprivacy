# Inf-Net Installation & Setup Guide

## Quick Start

This guide covers running Inf-Net for COVID-19 lung infection segmentation on A100 GPUs with support for multiple network architectures, differential privacy, and morphological operations.

## Prerequisites

- Linux system with A100 GPU access
- Python virtual environment (`.venv`)
- SLURM job scheduler
- PyTorch with CUDA support
- Opacus library (for differential privacy training)

## 1. Dataset Setup

### Dataset Structure

The Lung CT Infection Segmentation dataset should be organized as follows:

```
Dataset/
├── TrainingSet/
│   └── LungInfection-Train/
│       └── Doctor-label/
│           ├── Imgs/          # Training images (.jpg or .png)
│           ├── GT/            # Ground truth masks (.png)
│           └── Edge/          # Edge maps (.png)
└── TestingSet/
    └── LungInfection-Test/
        ├── Imgs/              # Test images (.jpg or .png)
        └── GT/                # Ground truth masks (.png)
```

### Download COVID-SemiSeg Dataset
```bash
# Download from Google Drive (manual)
# Link: https://drive.google.com/open?id=1bbKAqUuk7Y1q3xsDSwP07oOXN_GL3SQM

# Extract to Dataset directory
cd code/inf-net
unzip COVID-SemiSeg.zip
mv COVID-SemiSeg/Dataset/TrainingSet/* Dataset/TrainingSet/
mv COVID-SemiSeg/Dataset/TestingSet/* Dataset/TestingSet/
rm -rf COVID-SemiSeg COVID-SemiSeg.zip
```

### Download Pretrained Backbones
```bash
# Create directory
mkdir -p Snapshots/pre_trained

# Download backbone models
cd Snapshots/pre_trained
curl -L -o vgg16-397923af.pth https://download.pytorch.org/models/vgg16-397923af.pth
curl -L -o resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
curl -L -o res2net50_v1b_26w_4s-3cf99910.pth https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth
```

## 2. Training

The unified training script `MyTrain_LungInf_Unified.py` supports multiple configurations:

### Supported Networks
- **Inf_Net**: Original Inf-Net architecture with Res2Net50/ResNet50/VGGNet16 backbone
- **UNet**: UNet architecture with GroupNorm
- **NestedUNet**: NestedUNet (UNet++) architecture with GroupNorm

### Training Options

#### Basic Training (No DP, No Morphology)
```bash
python MyTrain_LungInf_Unified.py \
    --network Inf_Net \
    --epoch 70 \
    --batchsize 24 \
    --lr 1e-4 \
    --run 1 \
    --backbone Res2Net50
```

#### Training with Differential Privacy
```bash
python MyTrain_LungInf_Unified.py \
    --network Inf_Net \
    --epoch 70 \
    --batchsize 24 \
    --lr 1e-4 \
    --run 1 \
    --enable_privacy \
    --epsilon 8.0 \
    --max_grad_norm 1.2 \
    --clipping_strategy base \
    --backbone Res2Net50
```

**DP Parameters:**
- `--epsilon`: Privacy budget (common values: 8, 200)
- `--max_grad_norm`: Maximum gradient norm (common values: 1.2, 1.5, 2.0)
- `--clipping_strategy`: Clipping strategy (`base`, `automatic`, `psac`, `nsgd`)

#### Training with Morphology
```bash
python MyTrain_LungInf_Unified.py \
    --network Inf_Net \
    --epoch 70 \
    --batchsize 24 \
    --lr 1e-4 \
    --run 1 \
    --enable_morphology \
    --morph_operation both \
    --morph_kernel_size 3 \
    --backbone Res2Net50
```

**Morphology Parameters:**
- `--morph_operation`: Operation type (`open`, `close`, `both`, `dilation`, `erosion`)
- `--morph_kernel_size`: Kernel size (must be odd: 3, 5, 7, 9)

#### Training with DP + Morphology
```bash
python MyTrain_LungInf_Unified.py \
    --network Inf_Net \
    --epoch 70 \
    --batchsize 24 \
    --lr 1e-4 \
    --run 1 \
    --enable_privacy \
    --epsilon 8.0 \
    --max_grad_norm 1.2 \
    --clipping_strategy base \
    --enable_morphology \
    --morph_operation both \
    --morph_kernel_size 3 \
    --backbone Res2Net50
```

### Training via SLURM

Example SLURM training scripts are available in the `slurm/` directory. You can submit training jobs using:

```bash
# Example: Train Inf-Net base model
sbatch slurm/inf_net_base.sh

# Example: Train Inf-Net with DP
sbatch slurm/inf_net_dp_base.sh

# Example: Train Inf-Net with DP and Morphology
sbatch slurm/inf_net_dp_base_morph.sh
```

### Monitor Training

```bash
# Check job status
squeue -u $USER

# View training progress
tail -f logs/train_*.out

# Check for errors
tail -f logs/train_*.err
```

### Training Output

- **Models**: Saved to `Snapshots/save_weights/` with structured paths based on configuration
- **CSV Results**: Training metrics saved to `results/LungInfection/` and `results/all_results_training_global.csv`
- **Model Structure**: Models are saved with paths like:
  - Base: `{Network}_GroupNorm/batch_{batchsize}/run_{run}/{Network}-{epoch}.pth`
  - DP: `{Network}_DP/batch_{batchsize}/run_{run}/epsilon_{epsilon}/maxgrad_{max_grad_norm}/{clipping}/{Network}-{epoch}.pth`
  - Morph: `{Network}_Morph_GroupNorm/{operation}/kernel_{kernel_size}/batch_{batchsize}/run_{run}/{Network}-{epoch}.pth`

## 3. Testing/Inference

The testing script `MyTest_LungInf_All.py` supports single model testing and batch testing of all trained models.

### Single Model Testing

```bash
python MyTest_LungInf_All.py \
    --model_type Inf-Net_GroupNorm \
    --batchsize 24 \
    --run 1 \
    --epoch 70 \
    --testsize 352 \
    --data_path ./Dataset/TestingSet/LungInfection-Test/
```

### Testing DP Models

```bash
python MyTest_LungInf_All.py \
    --model_type Inf-Net_DP \
    --batchsize 24 \
    --run 1 \
    --epoch 70 \
    --epsilon 8.0 \
    --max_grad_norm 1.2 \
    --clipping_strategy base \
    --testsize 352 \
    --data_path ./Dataset/TestingSet/LungInfection-Test/
```

### Testing Morphology Models

```bash
python MyTest_LungInf_All.py \
    --model_type Inf-Net_Morph_GroupNorm \
    --batchsize 24 \
    --run 1 \
    --epoch 70 \
    --morph_operation both \
    --morph_kernel_size 3 \
    --testsize 352 \
    --data_path ./Dataset/TestingSet/LungInfection-Test/
```

### Batch Testing All Models

Test all final epoch models automatically:

```bash
# Test all final epoch models
python MyTest_LungInf_All.py \
    --test_all_final \
    --testsize 352 \
    --data_path ./Dataset/TestingSet/LungInfection-Test/

# Test only specific model types
python MyTest_LungInf_All.py \
    --test_all_final \
    --filter_model_type Inf-Net_DP \
    --testsize 352 \
    --data_path ./Dataset/TestingSet/LungInfection-Test/
```

### List Available Models

```bash
python MyTest_LungInf_All.py --list_models
```

### Testing Output

- **Predictions**: Saved to `Results/Lung_infection_segmentation/` with paths matching the model structure
- **Format**: PNG files with same names as input images
- **Structure**: Results organized by model type, batch size, run number, and configuration

## 4. Evaluation

The evaluation script computes metrics (Dice, IoU, etc.) for all predictions.

### Single Evaluation Run

```bash
# Submit evaluation job
sbatch MyEval_LungInf_All.sh
```

This evaluates all predictions in `Results/Lung_infection_segmentation/` against ground truth in `Dataset/TestingSet/LungInfection-Test/GT/`.

### Generate Evaluation Jobs

For large-scale evaluation, you can generate SLURM array jobs:

```bash
cd EvaluationToolPython
python ../legacy_archive/generate_eval_jobs.py \
    --results-dir ../Results/Lung_infection_segmentation \
    --gt-path ../Dataset/TestingSet/LungInfection-Test/GT/ \
    --output-dir slurm_jobs_eval

# Submit the generated array job
cd slurm_jobs_eval/full_run
sbatch array_job.sh
```

### Evaluation Output

- **Reports**: Saved to `EvaluateResults/Lung_infection_segmentation/`
- **Metrics**: Includes Dice coefficient, IoU, Sensitivity, Specificity, etc.
- **Format**: CSV files and summary reports organized by model configuration

## 5. Expected Results

### Training Output
- **Duration**: Varies by model and batch size (~30 minutes to 2 hours for 70 epochs)
- **Models**: Final checkpoint saved at epoch 70 in `Snapshots/save_weights/`
- **Loss**: Should decrease over training epochs
- **Privacy Budget**: For DP models, final epsilon is printed at the end of training

### Testing Output
- **Predictions**: 48 segmentation masks in `Results/Lung_infection_segmentation/`
- **Format**: PNG files with same names as input images
- **Organization**: Results organized by model type and configuration

### Evaluation Output
- **Metrics**: Comprehensive evaluation metrics for each model configuration
- **Reports**: Detailed CSV reports with per-image and aggregate statistics
- **Location**: `EvaluateResults/Lung_infection_segmentation/`

## 6. Model Types Reference

### Model Type Naming Convention

- **Base Models**: `Inf-Net_GroupNorm`, `UNet_GroupNorm`, `NestedUNet_GroupNorm`
- **DP Models**: `Inf-Net_DP`, `UNet_DP`, `NestedUNet_DP`
- **Morphology Models**: `Inf-Net_Morph_GroupNorm`, `UNet_Morph_GroupNorm`, `NestedUNet_Morph_GroupNorm`
- **DP + Morphology**: `Inf-Net_DP_Morph`, `UNet_DP_Morph`, `NestedUNet_DP_Morph`

### Common Training Configurations

| Configuration | Network | Batch Size | Epsilon | Max Grad Norm | Clipping | Morph |
|--------------|---------|------------|---------|---------------|----------|-------|
| Base | Inf_Net/UNet/NestedUNet | 24, 48 | - | - | - | - |
| DP Base | Inf_Net/UNet/NestedUNet | 24, 48 | 8, 200 | 1.2, 1.5, 2.0 | base, automatic, psac, nsgd | - |
| Morph | Inf_Net/UNet/NestedUNet | 24, 48 | - | - | - | both, open, close (kernel: 3, 5, 7, 9) |
| DP + Morph | Inf_Net/UNet/NestedUNet | 24, 48 | 8, 200 | 1.2, 1.5, 2.0 | base, automatic, psac, nsgd | both, open, close (kernel: 3, 5, 7, 9) |

## 7. Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**: Reduce batch size or use a smaller network
2. **Model Loading Errors**: Ensure model type matches training configuration
3. **Missing Dependencies**: Activate virtual environment before running scripts
4. **Path Errors**: Ensure dataset paths are correct relative to `code/inf-net/` directory

### Getting Help

- Check training/test logs in `logs/` directory
- Verify dataset structure matches expected format
- Ensure all pretrained backbones are downloaded
- Review model paths in `Snapshots/save_weights/` for correct structure
