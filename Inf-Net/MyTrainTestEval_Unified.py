#!/usr/bin/env python3
"""
Unified Training, Testing, Evaluation, and Aggregation Script for Lung Infection Segmentation

This script combines training, testing, evaluation, and aggregation into a single workflow
to make the process faster and more streamlined.

Usage:
    python MyTrainTestEval_Unified.py --network Inf_Net --batchsize 24 --run 1
    python MyTrainTestEval_Unified.py --network UNet --batchsize 48 --run 2 --enable_privacy --epsilon 8.0
    python MyTrainTestEval_Unified.py --network NestedUNet --enable_morphology --morph_operation both --morph_kernel_size 3

All training arguments are supported. After training completes, the script automatically:
1. Tests the trained model
2. Evaluates the test results
3. Aggregates metrics (merges training + evaluation, calculates mean ± std over runs)

The aggregation step updates the aggregated CSV files automatically, so you don't need to
run the aggregation script separately. Metrics are saved to:
- results/combined/aggregated_mean_std.csv (all models combined)
- results/aggregated/{Model}_aggregated_mean_std.csv (per-model files)

Append-only / no overwrite: training appends to the global training CSV; aggregation
merges with existing filtered/merged/aggregated CSVs so existing result rows are never
dropped (new data for the same experiment updates the row).
"""

import os
import sys
import subprocess
import argparse

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def determine_model_type_from_training_args(opt):
    """Determine model_type for testing from training arguments."""
    network_map = {
        "Inf_Net": "Inf-Net",
        "UNet": "UNet",
        "NestedUNet": "NestedUNet",
    }
    network_name = network_map.get(opt.network, opt.network)

    if opt.enable_privacy:
        if opt.enable_morphology:
            model_type = f"{network_name}_DP_Morph"
        else:
            model_type = f"{network_name}_DP"
    else:
        if opt.enable_morphology:
            model_type = f"{network_name}_Morph_GroupNorm"
        else:
            model_type = f"{network_name}_GroupNorm"

    return model_type


def build_result_dir_for_evaluation(opt, model_type):
    """Build the result directory path for evaluation from training args.

    This matches the structure used by MyTest_LungInf_All.py's build_result_path function.
    """
    if model_type in ["Inf-Net_GroupNorm", "UNet_GroupNorm", "NestedUNet_GroupNorm"]:
        result_dir = f"{model_type}/batch_{opt.batchsize}/run_{opt.run}"
    elif model_type in [
        "Inf-Net_Morph_GroupNorm",
        "UNet_Morph_GroupNorm",
        "NestedUNet_Morph_GroupNorm",
    ]:
        result_dir = f"{model_type}/{opt.morph_operation}/kernel_{opt.morph_kernel_size}/batch_{opt.batchsize}/run_{opt.run}"
    elif model_type in ["Inf-Net_DP", "UNet_DP", "NestedUNet_DP"]:
        clipping_dir = getattr(opt, "clipping_strategy", "base")
        if clipping_dir != "base":
            clipping_dir = clipping_dir
        else:
            clipping_dir = "base"
        epsilon = getattr(opt, "epsilon", 8.0)
        max_grad_norm = getattr(opt, "max_grad_norm", 1.2)
        result_dir = f"{model_type}/batch_{opt.batchsize}/run_{opt.run}/epsilon_{int(epsilon)}/maxgrad_{max_grad_norm}/{clipping_dir}"
    elif model_type in ["Inf-Net_DP_Morph", "UNet_DP_Morph", "NestedUNet_DP_Morph"]:
        clipping_dir = getattr(opt, "clipping_strategy", "base")
        if clipping_dir != "base":
            clipping_dir = clipping_dir
        else:
            clipping_dir = "base"
        epsilon = getattr(opt, "epsilon", 8.0)
        max_grad_norm = getattr(opt, "max_grad_norm", 1.2)
        result_dir = f"{model_type}/{opt.morph_operation}/kernel_{opt.morph_kernel_size}/batch_{opt.batchsize}/run_{opt.run}/epsilon_{int(epsilon)}/maxgrad_{max_grad_norm}/{clipping_dir}"
    else:
        result_dir = f"{model_type}/batch_{opt.batchsize}/run_{opt.run}"

    return result_dir


def run_training(opt):
    """Run training and return the training options object."""
    print("=" * 80)
    print("STEP 1: TRAINING")
    print("=" * 80)

    # Build training command
    train_script = os.path.join(os.path.dirname(__file__), "MyTrain_LungInf_Unified.py")

    train_args = [sys.executable, train_script]

    # Add all training arguments
    for key, value in vars(opt).items():
        # Skip arguments that are only for this unified script
        if key in [
            "test_data_path",
            "verbose_eval",
            "skip_training",
            "skip_testing",
            "skip_evaluation",
            "skip_aggregation",
        ]:
            continue

        if value is not None:
            if isinstance(value, bool):
                if value:
                    train_args.append(f"--{key}")
            else:
                train_args.append(f"--{key}")
                train_args.append(str(value))

    print(f"Running: {' '.join(train_args)}")

    # Run training
    result = subprocess.run(train_args, cwd=os.path.dirname(__file__))

    if result.returncode != 0:
        print(f"ERROR: Training failed with return code {result.returncode}")
        raise RuntimeError("Training failed")

    print("Training completed successfully!")
    return opt


def run_testing(opt, model_type):
    """Run testing on the trained model."""
    print("\n" + "=" * 80)
    print("STEP 2: TESTING")
    print("=" * 80)

    # Build test command
    test_script = os.path.join(os.path.dirname(__file__), "MyTest_LungInf_All.py")

    test_args = [
        sys.executable,
        test_script,
        "--model_type",
        model_type,
        "--batchsize",
        str(opt.batchsize),
        "--run",
        str(opt.run),
        "--epoch",
        str(opt.epoch),
        "--testsize",
        str(getattr(opt, "trainsize", 352)),
        "--data_path",
        getattr(opt, "test_data_path", "./Dataset/TestingSet/LungInfection-Test/"),
        "--gpu_device",
        str(opt.gpu_device),
    ]

    # Add DP parameters if needed
    if opt.enable_privacy:
        test_args.extend(["--epsilon", str(opt.epsilon)])
        test_args.extend(["--max_grad_norm", str(opt.max_grad_norm)])
        test_args.extend(["--clipping_strategy", opt.clipping_strategy])

    # Add morphology parameters if needed
    if opt.enable_morphology:
        test_args.extend(["--morph_operation", opt.morph_operation])
        test_args.extend(["--morph_kernel_size", str(opt.morph_kernel_size)])
        test_args.append("--enable_morphology_test")

    print(f"Running: {' '.join(test_args)}")

    # Run testing
    result = subprocess.run(test_args, cwd=os.path.dirname(__file__))

    if result.returncode != 0:
        print(f"ERROR: Testing failed with return code {result.returncode}")
        return None

    print("Testing completed successfully!")
    return True


def run_evaluation(opt, result_dir):
    """Run evaluation on the test results."""
    print("\n" + "=" * 80)
    print("STEP 3: EVALUATION")
    print("=" * 80)

    # Build evaluation command
    eval_script = os.path.join(
        os.path.dirname(__file__), "EvaluationToolPython", "main_all.py"
    )

    eval_args = [
        sys.executable,
        eval_script,
        "--gt_path",
        "../Dataset/TestingSet/LungInfection-Test/GT/",
        "--result_dir",
        result_dir,
    ]

    if getattr(opt, "verbose_eval", False):
        eval_args.append("--verbose")

    print(f"Running: {' '.join(eval_args)}")
    print(f"Evaluating result directory: {result_dir}")

    # Change to evaluation directory
    eval_dir = os.path.dirname(eval_script)

    # Run evaluation
    result = subprocess.run(eval_args, cwd=eval_dir)

    if result.returncode != 0:
        print(f"ERROR: Evaluation failed with return code {result.returncode}")
        return None

    print("Evaluation completed successfully!")
    return True


def run_aggregation(opt):
    """Run aggregation script to merge training and evaluation metrics."""
    print("\n" + "=" * 80)
    print("STEP 4: AGGREGATION")
    print("=" * 80)

    # Build aggregation command
    agg_script = os.path.join(os.path.dirname(__file__), "results", "agg_results.py")

    if not os.path.exists(agg_script):
        print(f"WARNING: Aggregation script not found at {agg_script}")
        print("Skipping aggregation step.")
        return None

    agg_args = [sys.executable, agg_script]

    print(f"Running: {' '.join(agg_args)}")
    print("Aggregating training and evaluation metrics...")
    print("This will update aggregated CSV files with mean ± std over runs.")

    # Run aggregation
    result = subprocess.run(agg_args, cwd=os.path.dirname(__file__))

    if result.returncode != 0:
        print(f"WARNING: Aggregation completed with return code {result.returncode}")
        print("This may be normal if not all runs are complete yet.")
        return None

    print("Aggregation completed successfully!")
    print("Aggregated metrics saved to:")
    print("  - results/combined/aggregated_mean_std.csv")
    print("  - results/aggregated/{Model}_aggregated_mean_std.csv")
    return True


def main():
    """Main function that orchestrates training, testing, and evaluation."""
    # Parse arguments (same as training script)
    parser = argparse.ArgumentParser(
        description="Unified Training, Testing, Evaluation, and Aggregation for Lung Infection Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs training, testing, evaluation, and aggregation in sequence.

Example usage:
    # Non-private baseline
    python MyTrainTestEval_Unified.py --network Inf_Net --batchsize 24 --run 1
    
    # DP training
    python MyTrainTestEval_Unified.py --network UNet --batchsize 48 --run 1 \\
        --enable_privacy --epsilon 8.0 --max_grad_norm 1.2 --clipping_strategy automatic
    
    # With morphology
    python MyTrainTestEval_Unified.py --network NestedUNet --batchsize 24 --run 1 \\
        --enable_morphology --morph_operation both --morph_kernel_size 3
    
After each experiment completes, metrics are automatically aggregated into CSV files
with mean ± std over runs. No need to run aggregation scripts separately!
        """,
    )

    # Network selection
    parser.add_argument(
        "--network",
        type=str,
        default="Inf_Net",
        choices=["Inf_Net", "UNet", "NestedUNet"],
        help="Network architecture to use",
    )

    # Hyper-parameters (same as training)
    parser.add_argument("--epoch", type=int, default=70, help="epoch number")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batchsize", type=int, default=24, help="training batch size")
    parser.add_argument(
        "--trainsize", type=int, default=352, help="set the size of training sample"
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=0.5,
        help="gradient clipping margin (not used with DP)",
    )
    parser.add_argument(
        "--decay_rate", type=float, default=0.1, help="decay rate of learning rate"
    )
    parser.add_argument(
        "--decay_epoch", type=int, default=50, help="every n epochs decay learning rate"
    )
    parser.add_argument(
        "--is_thop",
        type=bool,
        default=False,
        help="whether calculate FLOPs/Params (Thop)",
    )
    parser.add_argument(
        "--gpu_device",
        type=int,
        default=0,
        help="choose which GPU device you want to use",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="number of workers in dataloader",
    )

    # Model parameters
    parser.add_argument(
        "--net_channel",
        type=int,
        default=32,
        help="internal channel numbers in the Inf-Net, default=32",
    )
    parser.add_argument(
        "--init_features",
        type=int,
        default=32,
        help="initial feature channels in UNet",
    )
    parser.add_argument(
        "--n_classes", type=int, default=1, help="binary segmentation when n_classes=1"
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="input channels (3 for RGB CT images)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="Res2Net50",
        help="change different backbone, choice: VGGNet16, ResNet50, Res2Net50 (Inf_Net only)",
    )
    parser.add_argument(
        "--deep_supervision",
        type=bool,
        default=False,
        help="enable deep supervision for NestedUNet",
    )

    # Training dataset
    parser.add_argument(
        "--train_path",
        type=str,
        default="./Dataset/TrainingSet/LungInfection-Train/Doctor-label",
    )
    parser.add_argument(
        "--is_semi",
        type=bool,
        default=False,
        help="if True, you will turn on the mode of `Semi-Inf-Net`",
    )
    parser.add_argument(
        "--is_pseudo",
        type=bool,
        default=False,
        help="if True, you will train the model on pseudo-label",
    )
    parser.add_argument(
        "--train_save",
        type=str,
        default=None,
        help="If you use custom save path, please edit `--is_semi=True` and `--is_pseudo=True`",
    )
    parser.add_argument(
        "--run", type=int, default=1, help="the training iteration number"
    )

    # Privacy related arguments
    parser.add_argument(
        "--enable_privacy",
        action="store_true",
        help="Enable differential privacy training",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=8.0,
        help="Target privacy budget (epsilon) for DP-SGD. Common values: 8, 200",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.2,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Target privacy parameter (delta)",
    )
    parser.add_argument(
        "--clipping_strategy",
        type=str,
        default="base",
        choices=["base", "automatic", "psac", "nsgd"],
        help="Gradient clipping strategy for DP-SGD",
    )

    # Morphology arguments
    parser.add_argument(
        "--enable_morphology",
        action="store_true",
        help="Enable morphological operations during training",
    )
    parser.add_argument(
        "--morph_operation",
        type=str,
        default="both",
        choices=["open", "close", "dilation", "erosion", "both"],
        help="Morphological operation to apply",
    )
    parser.add_argument(
        "--morph_kernel_size",
        type=int,
        default=3,
        help="Morphological kernel size (must be odd)",
    )
    parser.add_argument(
        "--results_base",
        type=str,
        default="results",
        help="Base directory for results (default: results)",
    )

    # Testing parameters
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./Dataset/TestingSet/LungInfection-Test/",
        help="Path to test data",
    )

    # Evaluation parameters
    parser.add_argument(
        "--verbose_eval",
        action="store_true",
        help="Enable verbose output during evaluation",
    )

    # Skip steps (for debugging)
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training step (assume model already trained)",
    )
    parser.add_argument(
        "--skip_testing",
        action="store_true",
        help="Skip testing step (assume predictions already exist)",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation step",
    )
    parser.add_argument(
        "--skip_aggregation",
        action="store_true",
        help="Skip aggregation step (metrics will not be aggregated)",
    )

    opt = parser.parse_args()

    print("=" * 80)
    print("UNIFIED TRAINING, TESTING, EVALUATION, AND AGGREGATION")
    print("=" * 80)
    print(f"Network: {opt.network}")
    print(f"Batch Size: {opt.batchsize}")
    print(f"Run: {opt.run}")
    print(f"Epoch: {opt.epoch}")
    if opt.enable_privacy:
        print(
            f"DP Enabled: ε={opt.epsilon}, max_grad_norm={opt.max_grad_norm}, clipping={opt.clipping_strategy}"
        )
    if opt.enable_morphology:
        print(f"Morphology: {opt.morph_operation}, kernel_size={opt.morph_kernel_size}")
    print("=" * 80)

    # Determine model type for testing (needed even if skipping training)
    model_type = determine_model_type_from_training_args(opt)

    # Step 1: Training
    if not opt.skip_training:
        try:
            opt = run_training(opt)
            print("\n✓ Training completed successfully!")
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback

            traceback.print_exc()
            return 1
    else:
        print("\n[SKIPPING TRAINING]")
        print(f"Using model type: {model_type}")

    # Step 2: Testing
    if not opt.skip_testing:
        try:
            test_result = run_testing(opt, model_type)
            if test_result is None:
                print("\n✗ Testing failed!")
                return 1
            print("\n✓ Testing completed successfully!")
        except Exception as e:
            print(f"\n✗ Testing failed: {e}")
            import traceback

            traceback.print_exc()
            return 1
    else:
        print("\n[SKIPPING TESTING]")

    # Build result directory for evaluation
    result_dir = build_result_dir_for_evaluation(opt, model_type)
    print(f"\nResult directory for evaluation: {result_dir}")

    # Verify result directory exists before evaluation
    result_full_path = os.path.join(
        os.path.dirname(__file__), "Results", "Lung_infection_segmentation", result_dir
    )
    if not opt.skip_evaluation and not os.path.exists(result_full_path):
        print(f"\n⚠ WARNING: Result directory does not exist: {result_full_path}")
        print("Evaluation will be skipped. Check if testing completed successfully.")
        opt.skip_evaluation = True

    # Step 3: Evaluation
    if not opt.skip_evaluation:
        try:
            eval_result = run_evaluation(opt, result_dir)
            if eval_result is None:
                print("\n✗ Evaluation failed!")
                return 1
            print("\n✓ Evaluation completed successfully!")
        except Exception as e:
            print(f"\n✗ Evaluation failed: {e}")
            import traceback

            traceback.print_exc()
            return 1
    else:
        print("\n[SKIPPING EVALUATION]")

    # Step 4: Aggregation
    if not opt.skip_aggregation:
        try:
            agg_result = run_aggregation(opt)
            if agg_result is None:
                print("\n⚠ Aggregation completed with warnings (this may be normal)")
            else:
                print("\n✓ Aggregation completed successfully!")
        except Exception as e:
            print(f"\n⚠ Aggregation failed: {e}")
            print("This is non-critical - metrics are still saved individually.")
            import traceback

            traceback.print_exc()
    else:
        print("\n[SKIPPING AGGREGATION]")

    print("\n" + "=" * 80)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Training: Model saved to Snapshots/save_weights/...")
    print(
        f"Testing: Predictions saved to Results/Lung_infection_segmentation/{result_dir}"
    )
    print(
        f"Evaluation: Metrics saved to EvaluateResults/Lung_infection_segmentation/{result_dir}"
    )
    if not opt.skip_aggregation:
        print(
            f"Aggregation: Combined metrics saved to results/combined/aggregated_mean_std.csv"
        )
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
