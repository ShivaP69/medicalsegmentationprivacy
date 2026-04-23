#!/usr/bin/env python3
"""
Updated Evaluation tool for "Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans"
with support for new file structure

This is a Python implementation of the MATLAB evaluation code.
Author: Deng-Ping Fan, Tao Zhou, Ge-Peng Ji, Yi Zhou, Geng Chen, Huazhu Fu, Jianbing Shen, and Ling Shao
Updated by: Parth Shandilya (2025-01-03)

Metrics provided: Dice, IoU, F1, S-m (ICCV'17), E-m (IJCAI'18), Precision, Recall, Sensitivity, Specificity, MAE.

New features:
- Supports new file structure with batch sizes, runs, and model types
- Focuses on Lung_infection_segmentation with Inf-Net variants
- Uses LungInfection-Test dataset
- Saves results in organized evaluation results folder
"""

import os
import numpy as np
import cv2
import time
import glob
import argparse
from datetime import datetime

from enhanced_measure import enhanced_measure
from fmeasure_calu import fmeasure_calu
from structure_measure import structure_measure


def normalize_map(resmap):
    """Normalize resmap to [0, 1] range"""
    resmap_flat = resmap.flatten()
    min_val = np.min(resmap_flat)
    max_val = np.max(resmap_flat)
    if max_val > min_val:
        normalized = (resmap_flat - min_val) / (max_val - min_val)
    else:
        normalized = resmap_flat
    return normalized.reshape(resmap.shape)


def find_result_directories():
    """Find all result directories in the new structure"""
    # Match the underscore naming used by the testing scripts
    base_path = "../Results/Lung_infection_segmentation"
    result_dirs = []

    if not os.path.exists(base_path):
        print(f"Results directory not found: {base_path}")
        return result_dirs

    # Find all subdirectories
    for root, dirs, files in os.walk(base_path):
        # Check if this directory contains PNG files (test results)
        png_files = [f for f in files if f.endswith(".png")]
        if png_files:
            rel_path = os.path.relpath(root, base_path)
            result_dirs.append(
                {
                    "path": root,
                    "relative_path": rel_path,
                    "model_type": (
                        rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path
                    ),
                    "num_images": len(png_files),
                }
            )

    return result_dirs


def parse_result_path(relative_path):
    """Parse the relative path to extract model information (with new structure including kernel_size and max_grad_norm)"""
    parts = relative_path.split(os.sep)

    model_info = {
        "model_type": parts[0],
        "batch_size": None,
        "run": None,
        "epsilon": None,
        "max_grad_norm": None,
        "clipping_strategy": None,
        "morph_operation": None,
        "kernel_size": None,
    }

    # Parse based on model type structure
    # Non-DP models: model_type/batch_X/run_Y
    if parts[0] in ["Inf-Net", "Inf-Net_GroupNorm", "UNet_GroupNorm", "NestedUNet_GroupNorm"]:
        if len(parts) >= 3:
            model_info["batch_size"] = parts[1].replace("batch_", "")
            model_info["run"] = parts[2].replace("run_", "")

    # Non-DP with Morph: model_type/morph_op/kernel_{kernel_size}/batch_X/run_Y
    elif parts[0] in ["Inf-Net_Morph", "Inf-Net_Morph_GroupNorm", "UNet_Morph_GroupNorm", "NestedUNet_Morph_GroupNorm"]:
        if len(parts) >= 5:
            model_info["morph_operation"] = parts[1]
            model_info["kernel_size"] = parts[2].replace("kernel_", "")
            model_info["batch_size"] = parts[3].replace("batch_", "")
            model_info["run"] = parts[4].replace("run_", "")

    # DP models: model_type/batch_X/run_Y/epsilon_Z/maxgrad_{max_grad_norm}/clipping_strategy
    elif parts[0] in ["Inf-Net_DP", "UNet_DP", "NestedUNet_DP"]:
        if len(parts) >= 6:
            model_info["batch_size"] = parts[1].replace("batch_", "")
            model_info["run"] = parts[2].replace("run_", "")
            model_info["epsilon"] = parts[3].replace("epsilon_", "")
            model_info["max_grad_norm"] = parts[4].replace("maxgrad_", "")
            model_info["clipping_strategy"] = parts[5]

    # DP with Morph: model_type/morph_op/kernel_{kernel_size}/batch_X/run_Y/epsilon_Z/maxgrad_{max_grad_norm}/clipping_strategy
    elif parts[0] in ["Inf-Net_DP_Morph", "UNet_DP_Morph", "NestedUNet_DP_Morph"]:
        if len(parts) >= 8:
            model_info["morph_operation"] = parts[1]
            model_info["kernel_size"] = parts[2].replace("kernel_", "")
            model_info["batch_size"] = parts[3].replace("batch_", "")
            model_info["run"] = parts[4].replace("run_", "")
            model_info["epsilon"] = parts[5].replace("epsilon_", "")
            model_info["max_grad_norm"] = parts[6].replace("maxgrad_", "")
            model_info["clipping_strategy"] = parts[7]

    return model_info


def build_evaluation_result_path(model_info):
    """Build the evaluation result save path based on model info (matching new structure with kernel_size and max_grad_norm)"""
    base_path = "../EvaluateResults/Lung_infection_segmentation"

    # Non-DP models: model_type/batch_X/run_Y/
    if model_info["model_type"] in ["Inf-Net", "Inf-Net_GroupNorm", "UNet_GroupNorm", "NestedUNet_GroupNorm"]:
        result_path = os.path.join(
            base_path,
            model_info["model_type"],
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
        )
    # Non-DP with Morph: model_type/morph_op/kernel_{kernel_size}/batch_X/run_Y/
    elif model_info["model_type"] in ["Inf-Net_Morph", "Inf-Net_Morph_GroupNorm", "UNet_Morph_GroupNorm", "NestedUNet_Morph_GroupNorm"]:
        result_path = os.path.join(
            base_path,
            model_info["model_type"],
            model_info["morph_operation"],
            f"kernel_{model_info['kernel_size']}",
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
        )
    # DP models: model_type/batch_X/run_Y/epsilon_Z/maxgrad_{max_grad_norm}/clipping_strategy/
    elif model_info["model_type"] in ["Inf-Net_DP", "UNet_DP", "NestedUNet_DP"]:
        result_path = os.path.join(
            base_path,
            model_info["model_type"],
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
            f"epsilon_{model_info['epsilon']}",
            f"maxgrad_{model_info['max_grad_norm']}",
            model_info["clipping_strategy"],
        )
    # DP with Morph: model_type/morph_op/kernel_{kernel_size}/batch_X/run_Y/epsilon_Z/maxgrad_{max_grad_norm}/clipping_strategy/
    elif model_info["model_type"] in ["Inf-Net_DP_Morph", "UNet_DP_Morph", "NestedUNet_DP_Morph"]:
        result_path = os.path.join(
            base_path,
            model_info["model_type"],
            model_info["morph_operation"],
            f"kernel_{model_info['kernel_size']}",
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
            f"epsilon_{model_info['epsilon']}",
            f"maxgrad_{model_info['max_grad_norm']}",
            model_info["clipping_strategy"],
        )
    else:
        # Fallback for unknown model types
        result_path = os.path.join(base_path, model_info["model_type"])

    return result_path


def evaluate_single_model(result_dir_info, gt_path, opt):
    """Evaluate a single model's results"""
    model_info = parse_result_path(result_dir_info["relative_path"])
    result_map_path = result_dir_info["path"]

    print(f"\n{'='*80}")
    print(f"Evaluating: {model_info['model_type']}")
    if model_info["batch_size"]:
        print(f"Batch Size: {model_info['batch_size']}")
    if model_info["run"]:
        print(f"Run: {model_info['run']}")
    if model_info["epsilon"]:
        print(f"Epsilon: {model_info['epsilon']}")
    if model_info["max_grad_norm"]:
        print(f"Max Grad Norm: {model_info['max_grad_norm']}")
    if model_info["clipping_strategy"]:
        print(f"Clipping Strategy: {model_info['clipping_strategy']}")
    if model_info["morph_operation"]:
        print(f"Morph Operation: {model_info['morph_operation']}")
    if model_info["kernel_size"]:
        print(f"Kernel Size: {model_info['kernel_size']}")
    print(f"Result Path: {result_map_path}")
    print(f"Number of Images: {result_dir_info['num_images']}")
    print(f"{'='*80}")

    # Get list of image files
    img_files = glob.glob(os.path.join(result_map_path, "*.png"))
    img_num = len(img_files)

    if img_num == 0:
        print(f"No images found in {result_map_path}")
        return None

    # Initialize arrays for metrics
    thresholds = np.linspace(1, 0, 256)  # 1:-1/255:0 in MATLAB
    threshold_emeasure = np.zeros((img_num, len(thresholds)))
    threshold_precision = np.zeros((img_num, len(thresholds)))
    threshold_recall = np.zeros((img_num, len(thresholds)))
    threshold_sensitivity = np.zeros((img_num, len(thresholds)))
    threshold_specificity = np.zeros((img_num, len(thresholds)))
    threshold_dice = np.zeros((img_num, len(thresholds)))

    smeasure = np.zeros(img_num)
    mae = np.zeros(img_num)

    for i, img_file in enumerate(img_files):
        name = os.path.basename(img_file)
        if opt.verbose:
            print(f"Evaluating {name}: {i+1}/{img_num}")

        # Load ground truth
        gt_file = os.path.join(gt_path, name)
        if not os.path.exists(gt_file):
            print(f"Ground truth file not found: {gt_file}")
            continue

        gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"Could not load ground truth: {gt_file}")
            continue

        # Convert to logical
        gt = (gt > 128).astype(bool)

        # Load result map
        resmap = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if resmap is None:
            print(f"Could not load result map: {img_file}")
            continue

        # Check size
        if resmap.shape != gt.shape:
            resmap = cv2.resize(resmap, (gt.shape[1], gt.shape[0]))
            if opt.verbose:
                print(f"Resized result map to match ground truth: {img_file}")

        resmap = resmap.astype(np.float64) / 255.0

        # Normalize resmap to [0, 1]
        resmap = normalize_map(resmap)

        # S-measure metric published in ICCV'17
        smeasure[i] = structure_measure(resmap, gt)

        # Calculate threshold-based metrics
        threshold_e = np.zeros(len(thresholds))
        threshold_pr = np.zeros(len(thresholds))
        threshold_rec = np.zeros(len(thresholds))
        threshold_spe = np.zeros(len(thresholds))
        threshold_dic = np.zeros(len(thresholds))

        for t, threshold in enumerate(thresholds):
            precision, recall, specificity, dice, fmeasure = fmeasure_calu(
                resmap, gt.astype(float), gt.shape, threshold
            )

            threshold_pr[t] = precision
            threshold_rec[t] = recall
            threshold_spe[t] = specificity
            threshold_dic[t] = dice

            # Create binary resmap for E-measure
            bi_resmap = np.zeros_like(resmap)
            bi_resmap[resmap >= threshold] = 1
            threshold_e[t] = enhanced_measure(bi_resmap, gt)

        threshold_emeasure[i, :] = threshold_e
        threshold_sensitivity[i, :] = threshold_rec
        threshold_specificity[i, :] = threshold_spe
        threshold_dice[i, :] = threshold_dic

        # Calculate MAE
        mae[i] = np.mean(np.abs(gt.astype(float) - resmap))

    # Calculate final metrics
    mae_final = np.mean(mae)

    # Sensitivity
    column_sen = np.mean(threshold_sensitivity, axis=0)
    mean_sen = np.mean(column_sen)
    max_sen = np.max(column_sen)

    # Specificity
    column_spe = np.mean(threshold_specificity, axis=0)
    mean_spe = np.mean(column_spe)
    max_spe = np.max(column_spe)

    # Dice
    column_dic = np.mean(threshold_dice, axis=0)
    mean_dic = np.mean(column_dic)
    max_dic = np.max(column_dic)

    # E-m
    column_e = np.mean(threshold_emeasure, axis=0)
    mean_em = np.mean(column_e)
    max_em = np.max(column_e)

    # Sm
    sm = np.mean(smeasure)

    # Build evaluation result path
    eval_result_path = build_evaluation_result_path(model_info)
    os.makedirs(eval_result_path, exist_ok=True)

    # Create model name for saving
    model_name = model_info["model_type"]
    if model_info["batch_size"]:
        model_name += f"_batch{model_info['batch_size']}"
    if model_info["run"]:
        model_name += f"_run{model_info['run']}"
    if model_info["epsilon"]:
        model_name += f"_eps{model_info['epsilon']}"
    if model_info["max_grad_norm"]:
        model_name += f"_mg{model_info['max_grad_norm']}"
    if model_info["clipping_strategy"]:
        model_name += f"_{model_info['clipping_strategy']}"
    if model_info["morph_operation"]:
        model_name += f"_{model_info['morph_operation']}"
    if model_info["kernel_size"]:
        model_name += f"_k{model_info['kernel_size']}"

    # Save detailed results
    np.savez(
        os.path.join(eval_result_path, f"{model_name}.npz"),
        Sm=sm,
        mae=mae_final,
        column_Dic=column_dic,
        column_Sen=column_sen,
        column_Spe=column_spe,
        column_E=column_e,
        maxDic=max_dic,
        maxEm=max_em,
        maxSen=max_sen,
        maxSpe=max_spe,
        meanDic=mean_dic,
        meanEm=mean_em,
        meanSen=mean_sen,
        meanSpe=mean_spe,
    )

    # Save summary results (individual model summary)
    summary_file = os.path.join(eval_result_path, f"{model_name}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Model: {model_info['model_type']}\n")
        if model_info["batch_size"]:
            f.write(f"Batch Size: {model_info['batch_size']}\n")
        if model_info["run"]:
            f.write(f"Run: {model_info['run']}\n")
        if model_info["epsilon"]:
            f.write(f"Epsilon: {model_info['epsilon']}\n")
        if model_info["max_grad_norm"]:
            f.write(f"Max Grad Norm: {model_info['max_grad_norm']}\n")
        if model_info["clipping_strategy"]:
            f.write(f"Clipping Strategy: {model_info['clipping_strategy']}\n")
        if model_info["morph_operation"]:
            f.write(f"Morph Operation: {model_info['morph_operation']}\n")
        if model_info["kernel_size"]:
            f.write(f"Kernel Size: {model_info['kernel_size']}\n")
        f.write(f"Number of Images: {img_num}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nResults:\n")
        f.write(f"Mean Dice: {mean_dic:.4f}\n")
        f.write(f"Max Dice: {max_dic:.4f}\n")
        f.write(f"Mean Sensitivity: {mean_sen:.4f}\n")
        f.write(f"Max Sensitivity: {max_sen:.4f}\n")
        f.write(f"Mean Specificity: {mean_spe:.4f}\n")
        f.write(f"Max Specificity: {max_spe:.4f}\n")
        f.write(f"S-measure: {sm:.4f}\n")
        f.write(f"Mean E-measure: {mean_em:.4f}\n")
        f.write(f"Max E-measure: {max_em:.4f}\n")
        f.write(f"MAE: {mae_final:.4f}\n")

    # Print results
    result_line = f"meanDic:{mean_dic:.3f}; meanSen:{mean_sen:.3f}; meanSpe:{mean_spe:.3f}; Sm:{sm:.3f}; meanEm:{mean_em:.3f}; MAE:{mae_final:.3f}"
    print(f"Results: {result_line}")

    return {
        "model_info": model_info,
        "model_name": model_name,
        "mean_dic": mean_dic,
        "max_dic": max_dic,
        "mean_sen": mean_sen,
        "max_sen": max_sen,
        "mean_spe": mean_spe,
        "max_spe": max_spe,
        "sm": sm,
        "mean_em": mean_em,
        "max_em": max_em,
        "mae": mae_final,
        "num_images": img_num,
        "eval_path": eval_result_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Inf-Net models with new file structure"
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="../Dataset/TestingSet/LungInfection-Test/GT/",
        help="Path to ground truth images",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--model_filter",
        type=str,
        default=None,
        help='Filter models by type (e.g., "Inf-Net", "Inf-Net_DP", "Inf-Net_Morph")',
    )
    parser.add_argument(
        "--batch_filter",
        type=str,
        default=None,
        help='Filter by batch size (e.g., "32", "64", "128")',
    )
    parser.add_argument(
        "--run_filter",
        type=str,
        default=None,
        help='Filter by run number (e.g., "1", "2", "3")',
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Evaluate a single result directory (relative to base Results path)",
    )

    opt = parser.parse_args()

    print("=" * 100)
    print("Inf-Net Evaluation Tool (New Structure)")
    print("=" * 100)
    print(f"Ground Truth Path: {opt.gt_path}")
    print(f"Verbose: {opt.verbose}")
    if opt.model_filter:
        print(f"Model Filter: {opt.model_filter}")
    if opt.batch_filter:
        print(f"Batch Filter: {opt.batch_filter}")
    if opt.run_filter:
        print(f"Run Filter: {opt.run_filter}")
    print("=" * 100)

    # Check if ground truth path exists
    if not os.path.exists(opt.gt_path):
        print(f"Error: Ground truth path not found: {opt.gt_path}")
        return

    # Find all result directories
    print("Scanning for result directories...")
    result_dirs = find_result_directories()

    if not result_dirs:
        print("No result directories found!")
        return

    print(f"Found {len(result_dirs)} result directories")

    # If --result_dir is specified, evaluate only that directory
    if opt.result_dir:
        base_path = "../Results/Lung_infection_segmentation"
        target_path = os.path.join(base_path, opt.result_dir)
        if not os.path.exists(target_path):
            print(f"Error: Result directory not found: {target_path}")
            return
        
        # Find matching directory
        filtered_dirs = [d for d in result_dirs if d["relative_path"] == opt.result_dir]
        if not filtered_dirs:
            print(f"Error: Result directory not found in scan: {opt.result_dir}")
            return
        print(f"Evaluating single directory: {opt.result_dir}")
    else:
        # Apply filters
        filtered_dirs = result_dirs
    if opt.model_filter:
        filtered_dirs = [
            d for d in filtered_dirs if opt.model_filter in d["relative_path"]
        ]
        print(f"After model filter: {len(filtered_dirs)} directories")

    if opt.batch_filter:
        filtered_dirs = [
            d
            for d in filtered_dirs
            if f"batch_{opt.batch_filter}" in d["relative_path"]
        ]
        print(f"After batch filter: {len(filtered_dirs)} directories")

    if opt.run_filter:
        filtered_dirs = [
            d for d in filtered_dirs if f"run_{opt.run_filter}" in d["relative_path"]
        ]
        print(f"After run filter: {len(filtered_dirs)} directories")

    if not filtered_dirs:
        print("No directories match the filters!")
        return

    # Evaluate each model
    start_time = time.time()
    results = []

    for i, result_dir in enumerate(filtered_dirs, 1):
        print(f"\n[{i}/{len(filtered_dirs)}] Processing: {result_dir['relative_path']}")
        try:
            result = evaluate_single_model(result_dir, opt.gt_path, opt)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error evaluating {result_dir['relative_path']}: {str(e)}")

    # Print overall summary
    if results:
        print("\n" + "=" * 100)
        print("EVALUATION SUMMARY")
        print("=" * 100)

        # Sort by mean Dice score
        results.sort(key=lambda x: x["mean_dic"], reverse=True)

        print(
            f"{'Rank':<4} {'Model':<30} {'Mean Dice':<10} {'S-measure':<10} {'MAE':<8} {'Images':<7}"
        )
        print("-" * 100)

        for i, result in enumerate(results, 1):
            model_name = result["model_name"]
            if len(model_name) > 30:
                model_name = model_name[:27] + "..."

            print(
                f"{i:<4} {model_name:<30} {result['mean_dic']:<10.4f} {result['sm']:<10.4f} {result['mae']:<8.4f} {result['num_images']:<7}"
            )

        # Save/update overall summary (merge with existing to preserve previous evaluations)
        summary_file = (
            "../EvaluateResults/Lung_infection_segmentation/evaluation_summary.txt"
        )
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)

        # Read existing summary if it exists to preserve previous entries
        existing_results = {}
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                lines = f.readlines()
                # Parse existing entries (skip header lines)
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("Inf-Net") or line.startswith("Last Updated") or line.startswith("Total") or line.startswith("Ground") or line.startswith("Results") or line.startswith("Rank") or line.startswith("-"):
                        continue
                    # Look for result lines (format: "Rank Model MeanDice MaxDice S-measure MAE Images")
                    parts = line.split()
                    if len(parts) >= 7 and parts[0].isdigit():
                        try:
                            # Format: rank model_name mean_dice max_dice s_measure mae images
                            # Model name might have underscores, so we need to be careful
                            # The model name is everything between rank and the numeric fields
                            # Find where numeric fields start (after model name)
                            numeric_start = None
                            for i in range(1, len(parts)):
                                try:
                                    float(parts[i])
                                    if numeric_start is None:
                                        numeric_start = i
                                except ValueError:
                                    pass
                            
                            if numeric_start and numeric_start >= 2:
                                model_name = " ".join(parts[1:numeric_start])
                                if len(parts) >= numeric_start + 5:
                                    existing_results[model_name] = {
                                        'mean_dic': float(parts[numeric_start]),
                                        'max_dic': float(parts[numeric_start + 1]),
                                        'sm': float(parts[numeric_start + 2]),
                                        'mae': float(parts[numeric_start + 3]),
                                        'num_images': int(parts[numeric_start + 4]),
                                    }
                        except (ValueError, IndexError) as e:
                            # Skip malformed lines
                            continue

        # Update with new results (new results override old ones for same model)
        for result in results:
            existing_results[result['model_name']] = {
                'mean_dic': result['mean_dic'],
                'max_dic': result['max_dic'],
                'sm': result['sm'],
                'mae': result['mae'],
                'num_images': result['num_images'],
            }

        # Convert to list and sort by mean dice
        all_results_list = [
            {
                'model_name': name,
                **metrics
            }
            for name, metrics in existing_results.items()
        ]
        all_results_list.sort(key=lambda x: x['mean_dic'], reverse=True)

        # Write updated summary
        with open(summary_file, "w") as f:
            f.write(f"Inf-Net Evaluation Summary\n")
            f.write(
                f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total Models Evaluated: {len(all_results_list)}\n")
            f.write(f"Ground Truth Path: {opt.gt_path}\n")
            f.write(f"\nResults (sorted by Mean Dice):\n")
            f.write(
                f"{'Rank':<4} {'Model':<40} {'Mean Dice':<10} {'Max Dice':<10} {'S-measure':<10} {'MAE':<8} {'Images':<7}\n"
            )
            f.write("-" * 100 + "\n")

            for i, result in enumerate(all_results_list, 1):
                f.write(
                    f"{i:<4} {result['model_name']:<40} {result['mean_dic']:<10.4f} {result['max_dic']:<10.4f} {result['sm']:<10.4f} {result['mae']:<8.4f} {result['num_images']:<7}\n"
                )

        print(f"\nOverall summary updated at: {summary_file}")
        print(f"  Total entries: {len(all_results_list)} (including {len(existing_results) - len(results)} previous entries)")

    elapsed_time = time.time() - start_time
    print(f"\nTotal evaluation time: {elapsed_time:.2f} seconds")
    print(f"Successfully evaluated {len(results)} models")


if __name__ == "__main__":
    main()
