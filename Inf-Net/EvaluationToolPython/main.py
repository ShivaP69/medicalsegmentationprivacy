#!/usr/bin/env python3
"""
Evaluation tool for "Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans"

This is a Python implementation of the MATLAB evaluation code.
Author: Deng-Ping Fan, Tao Zhou, Ge-Peng Ji, Yi Zhou, Geng Chen, Huazhu Fu, Jianbing Shen, and Ling Shao
Homepage: http://dpfan.net/
Projectpage: https://github.com/DengPingFan/Inf-Net

Metrics provided: Dice, IoU, F1, S-m (ICCV'17), E-m (IJCAI'18), Precision, Recall, Sensitivity, Specificity, MAE.
"""

import os
import numpy as np
import cv2
import time
import glob

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


def main():
    # ---- 1. ResultMap Path Setting ----
    result_map_path = "../Results/"
    results = ["Lung_infection_segmentation", "Multi-class lung infection segmentation"]

    models_lung_inf = ["UNet", "UNet++", "Inf-Net", "Semi-Inf-Net"]
    models_multiclass_lung_inf = [
        "DeepLabV3Plus_Stride8",
        "DeepLabV3Plus_Stride16",
        "FCN8s_1100",
        "Semi-Inf-Net_FCN8s_1100",
        "Semi-Inf-Net_UNet",
    ]

    multiclass = ["Ground-glass opacities", "Consolidation"]

    # ---- 2. Ground-truth Datasets Setting ----
    data_path = "../Dataset/TestingSet/"
    datasets = ["LungInfection-Test", "MultiClassInfection-Test"]

    # ---- 3. Evaluation Results Save Path Setting ----
    res_dir = "../EvaluateResults/" # Add run number here

    res_name = "_result.txt"  # You can change the result name

    thresholds = np.linspace(1, 0, 256)  # 1:-1/255:0 in MATLAB
    dataset_num = len(datasets)

    for d in range(dataset_num):
        start_time = time.time()
        dataset = datasets[d]
        print(f"Processing {d+1}/{dataset_num}: {dataset} Dataset")

        if d == 0:
            cur_model = models_lung_inf
            num_folder = 1
        else:
            cur_model = models_multiclass_lung_inf
            num_folder = 2

        # For Lung infection segmentation, there is only one folder;
        # For Multi-class lung infection segmentation, there are two folders.
        for c in range(num_folder):
            model_num = len(cur_model)

            res_path = os.path.join(res_dir, f"{dataset}-{c+1}-mat/")
            if not os.path.exists(res_path):
                os.makedirs(res_path)

            res_txt = os.path.join(res_dir, f"{dataset}-{c+1}{res_name}")

            with open(res_txt, "w") as file_id:
                for m in range(model_num):
                    model = cur_model[m]
                    print(f"Processing model: {model}")

                    if d == 0:
                        gt_path = os.path.join(data_path, dataset, "GT/")
                        res_map_path = os.path.join(
                            result_map_path, results[d], model + "/" # Add run number here
                        )
                    else:
                        gt_path = os.path.join(data_path, dataset, f"GT-{c+1}/")
                        res_map_path = os.path.join(
                            result_map_path, results[d], multiclass[c], model + "/" # Add run number here
                        )

                    # Get list of image files
                    img_files = glob.glob(os.path.join(res_map_path, "*.png"))
                    img_num = len(img_files)

                    if img_num == 0:
                        print(f"No images found in {res_map_path}")
                        continue

                    # Initialize arrays for metrics
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
                        print(
                            f"Evaluating({dataset} Dataset, {model} Model, {name} Image): {i+1}/{img_num}"
                        )

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
                            cv2.imwrite(img_file, resmap)
                            print(
                                f"Resizing have been operated!! The resmap size does not match gt in the path: {img_file}"
                            )

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
                            precision, recall, specificity, dice, fmeasure = (
                                fmeasure_calu(
                                    resmap, gt.astype(float), gt.shape, threshold
                                )
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

                    # Save results
                    np.savez(
                        os.path.join(res_path, model + ".npz"),
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

                    # Write to file and print results
                    result_line = f"(Dataset:{dataset}; Model:{model}) meanDic:{mean_dic:.3f};meanSen:{mean_sen:.3f};meanSpe:{mean_spe:.3f};Sm:{sm:.3f};meanEm:{mean_em:.3f};MAE:{mae_final:.3f}.\n"
                    file_id.write(result_line)
                    print(result_line.strip())

        elapsed_time = time.time() - start_time
        print(f"Dataset {dataset} completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
