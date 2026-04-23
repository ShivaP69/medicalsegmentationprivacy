# Evaluation Tool in Python

This project is a Python implementation of various evaluation metrics originally developed in MATLAB. It includes several modules that provide functionality for calculating different metrics used in model evaluation.

## Project Structure

```
EvaluationToolPython
├── src
│   ├── calmae.py               # Implementation of the CalMAE metric
│   ├── enhanced_measure.py      # Enhanced measurement techniques
│   ├── fmeasure_calu.py         # Calculation of the F-measure
│   ├── main.py                  # Entry point of the application
│   ├── original_wfb.py          # Original weighted feature-based metric
│   ├── s_object.py              # Object-related structures and methods
│   ├── s_region.py              # Region-related structures and methods
│   └── structure_measure.py      # Structural measures for evaluation
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation
```
## Expected Directory Structure

The tool expects the following directory structure relative to the main script:

```
../Results/                    # Result maps from your models
├── Lung infection segmentation/
│   ├── UNet/
│   ├── UNet++/
│   ├── Inf-Net/
│   └── Semi-Inf-Net/
└── Multi-class lung infection segmentation/
    ├── Ground-glass opacities/
    └── Consolidation/

../Dataset/TestingSet/         # Ground truth datasets
├── LungInfection-Test/
│   └── GT/
└── MultiClassInfection-Test/
    ├── GT-1/
    └── GT-2/

../EvaluateResults/            # Output directory for results
```

## Metrics Description

- **MAE**: Mean Absolute Error between prediction and ground truth
- **Dice**: Dice similarity coefficient (F1-score for binary classification)
- **Sensitivity/Recall**: True positive rate
- **Specificity**: True negative rate  
- **S-measure**: Structure-measure evaluating region-aware and object-aware similarities
- **E-measure**: Enhanced-alignment measure capturing pixel-level matching and image-level statistics
- **Precision**: Positive predictive value

## Usage

To run the evaluation metrics, execute the `main.py` file:

```bash
python src/main.py
```

## Output

The tool generates:
- Text files with evaluation results in `../EvaluateResults/`
- NumPy archives (.npz) with detailed metric arrays
- Console output showing progress and final results
