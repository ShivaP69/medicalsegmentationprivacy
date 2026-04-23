# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
with support for new file structure and different model types
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
Updated Version: Support for new file structure on 2025-10-23 (@author: Parth Shandilya)
"""

import torch
import numpy as np
import os
import argparse
from PIL import Image
import glob
from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from Code.utils.dataloader_LungInf import test_dataset

# Differential Privacy
from opacus.validators import ModuleValidator


def clean_state_dict(state_dict):
    """Clean a loaded state_dict by removing THOP keys and DataParallel prefixes.

    - Drops keys ending with total_ops/total_params (from thop)
    - Strips common prefixes added by wrappers like DataParallel
    """
    filtered_state_dict = {}
    for k, v in state_dict.items():
        # Skip THOP keys
        if k.endswith(("total_ops", "total_params")):
            continue

        # Remove common wrapper prefixes if present
        if k.startswith("module."):
            new_key = k[len("module.") :]
        elif k.startswith("_module."):
            new_key = k[len("_module.") :]
        else:
            new_key = k

        filtered_state_dict[new_key] = v

    return filtered_state_dict


def load_model_for_inference(model_path, model_type, device="cuda"):
    """
    Load model with correct architecture based on training type

    Args:
        model_path: Path to checkpoint file
        model_type: One of 'Inf-Net', 'Inf-Net_Morph', 'Inf-Net_DP', 'Inf-Net_DP_Morph',
                    'Inf-Net_GroupNorm', 'Inf-Net_Morph_GroupNorm',
                    'UNet_GroupNorm', 'UNet_Morph_GroupNorm', 'UNet_DP', 'UNet_DP_Morph',
                    'NestedUNet_GroupNorm', 'NestedUNet_Morph_GroupNorm', 'NestedUNet_DP', 'NestedUNet_DP_Morph'
        device: Device to load model on

    Returns:
        model: Loaded model with correct architecture
    """
    # Create base model based on type
    if "UNet" in model_type and "Nested" not in model_type:
        # UNet model
        from Code.model_lung_infection.InfNet_UNet_GroupNorm import UNet_GroupNorm

        model = UNet_GroupNorm(
            in_channels=3,
            out_channels=1,
            init_features=32,
        )
        print("Loading UNet architecture")
    elif "NestedUNet" in model_type:
        # NestedUNet model
        from Code.model_lung_infection.InfNet_NestedUNet_GroupNorm import (
            NestedUNet_GroupNorm,
        )

        model = NestedUNet_GroupNorm(
            input_channels=3,
            num_classes=1,
            deep_supervision=False,
        )
        print("Loading NestedUNet architecture")
    else:
        # Inf-Net model (default)
        model = Network()
        print("Loading Inf-Net architecture")

    # All models in this codebase use GroupNorm (for DP compatibility or fair comparison)
    # If model was trained with DP or GroupNorm, convert architecture to GroupNorm
    is_groupnorm_model = "DP" in model_type or "GroupNorm" in model_type

    if is_groupnorm_model:
        print(
            "GroupNorm model detected, Converting BatchNorm to GroupNorm for matching training"
        )

        # Apply the same conversion that was done during training
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
            print("Model architecture converted to GroupNorm")
        else:
            print("Model already GroupNorm-compatible")
    else:
        print("Non-GroupNorm model, Using original BatchNorm architecture")

    # Move to device
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)

    # Clean state dict (remove THOP keys, wrapper prefixes)
    filtered_state_dict = clean_state_dict(state_dict)

    # Load with strict=True (should work now that architectures match)
    try:
        model.load_state_dict(filtered_state_dict, strict=True)
        print("Checkpoint loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"Warning: Could not load with strict=True: {e}")
        print("Falling back to strict=False (may cause poor performance)")
        model.load_state_dict(filtered_state_dict, strict=False)

    # Set to evaluation mode
    model.eval()

    return model


def build_model_path(opt):
    """Build the model path based on configuration (matching snapshot structure)"""
    base_path = "./Snapshots/save_weights"

    # Map model types to network names and checkpoint names
    network_map = {
        "Inf-Net": ("Inf-Net", "Inf-Net"),
        "Inf-Net_Morph": ("Inf-Net", "Inf-Net"),
        "Inf-Net_GroupNorm": ("Inf-Net", "Inf-Net"),
        "Inf-Net_Morph_GroupNorm": ("Inf-Net", "Inf-Net"),
        "Inf-Net_DP": ("Inf-Net", "Inf-Net"),
        "Inf-Net_DP_Morph": ("Inf-Net", "Inf-Net"),
        "UNet_GroupNorm": ("UNet", "UNet"),
        "UNet_Morph_GroupNorm": ("UNet", "UNet"),
        "UNet_DP": ("UNet", "UNet"),
        "UNet_DP_Morph": ("UNet", "UNet"),
        "NestedUNet_GroupNorm": ("NestedUNet", "NestedUNet"),
        "NestedUNet_Morph_GroupNorm": ("NestedUNet", "NestedUNet"),
        "NestedUNet_DP": ("NestedUNet", "NestedUNet"),
        "NestedUNet_DP_Morph": ("NestedUNet", "NestedUNet"),
    }

    network_name, checkpoint_name = network_map.get(opt.model_type, (None, None))

    if opt.model_type == "Inf-Net":
        # Structure: Inf-Net/batch_X/run_Y/Inf-Net-Z.pth
        model_path = os.path.join(
            base_path,
            "Inf-Net",
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
            f"Inf-Net-{opt.epoch}.pth",
        )
    elif opt.model_type in [
        "Inf-Net_GroupNorm",
        "UNet_GroupNorm",
        "NestedUNet_GroupNorm",
    ]:
        # Structure: {Network}_GroupNorm/batch_X/run_Y/{Network}-Z.pth
        model_path = os.path.join(
            base_path,
            opt.model_type,
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
            f"{checkpoint_name}-{opt.epoch}.pth",
        )
    elif opt.model_type in [
        "Inf-Net_Morph_GroupNorm",
        "UNet_Morph_GroupNorm",
        "NestedUNet_Morph_GroupNorm",
    ]:
        # Structure: {Network}_Morph_GroupNorm/{operation}/kernel_{kernel_size}/batch_X/run_Y/{Network}-Z.pth
        model_path = os.path.join(
            base_path,
            opt.model_type,
            opt.morph_operation,
            f"kernel_{opt.morph_kernel_size}",
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
            f"{checkpoint_name}-{opt.epoch}.pth",
        )
    elif opt.model_type in ["Inf-Net_DP", "UNet_DP", "NestedUNet_DP"]:
        # Structure: {Network}_DP/batch_X/run_Y/epsilon_Z/maxgrad_{max_grad_norm}/{clipping}/{Network}-E.pth
        clipping_dir = (
            opt.clipping_strategy if opt.clipping_strategy != "base" else "base"
        )
        model_path = os.path.join(
            base_path,
            opt.model_type,
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
            f"epsilon_{int(opt.epsilon)}",
            f"maxgrad_{opt.max_grad_norm}",
            clipping_dir,
            f"{checkpoint_name}-{opt.epoch}.pth",
        )
    elif opt.model_type in ["Inf-Net_DP_Morph", "UNet_DP_Morph", "NestedUNet_DP_Morph"]:
        # Structure: {Network}_DP_Morph/{operation}/kernel_{kernel_size}/batch_X/run_Y/epsilon_Z/maxgrad_{max_grad_norm}/{clipping}/{Network}-E.pth
        clipping_dir = (
            opt.clipping_strategy if opt.clipping_strategy != "base" else "base"
        )
        model_path = os.path.join(
            base_path,
            opt.model_type,
            opt.morph_operation,
            f"kernel_{opt.morph_kernel_size}",
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
            f"epsilon_{int(opt.epsilon)}",
            f"maxgrad_{opt.max_grad_norm}",
            clipping_dir,
            f"{checkpoint_name}-{opt.epoch}.pth",
        )
    else:
        # Custom path
        model_path = opt.pth_path

    return model_path


def build_result_path(opt):
    """Build the result save path based on configuration (matching snapshot structure)"""
    base_path = "./Results/Lung_infection_segmentation"

    if opt.model_type == "Inf-Net":
        # Structure: Inf-Net/batch_X/run_Y/
        result_path = os.path.join(
            base_path, "Inf-Net", f"batch_{opt.batchsize}", f"run_{opt.run}"
        )
    elif opt.model_type in [
        "Inf-Net_GroupNorm",
        "UNet_GroupNorm",
        "NestedUNet_GroupNorm",
    ]:
        # Structure: {Network}_GroupNorm/batch_X/run_Y/
        result_path = os.path.join(
            base_path, opt.model_type, f"batch_{opt.batchsize}", f"run_{opt.run}"
        )
    elif opt.model_type in [
        "Inf-Net_Morph_GroupNorm",
        "UNet_Morph_GroupNorm",
        "NestedUNet_Morph_GroupNorm",
    ]:
        # Structure: {Network}_Morph_GroupNorm/{operation}/kernel_{kernel_size}/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            opt.model_type,
            opt.morph_operation,
            f"kernel_{opt.morph_kernel_size}",
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
        )
    elif opt.model_type in ["Inf-Net_DP", "UNet_DP", "NestedUNet_DP"]:
        # Structure: {Network}_DP/batch_X/run_Y/epsilon_Z/maxgrad_{max_grad_norm}/{clipping}/
        clipping_dir = (
            opt.clipping_strategy if opt.clipping_strategy != "base" else "base"
        )
        result_path = os.path.join(
            base_path,
            opt.model_type,
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
            f"epsilon_{int(opt.epsilon)}",
            f"maxgrad_{opt.max_grad_norm}",
            clipping_dir,
        )
    elif opt.model_type in ["Inf-Net_DP_Morph", "UNet_DP_Morph", "NestedUNet_DP_Morph"]:
        # Structure: {Network}_DP_Morph/{operation}/kernel_{kernel_size}/batch_X/run_Y/epsilon_Z/maxgrad_{max_grad_norm}/{clipping}/
        clipping_dir = (
            opt.clipping_strategy if opt.clipping_strategy != "base" else "base"
        )
        result_path = os.path.join(
            base_path,
            opt.model_type,
            opt.morph_operation,
            f"kernel_{opt.morph_kernel_size}",
            f"batch_{opt.batchsize}",
            f"run_{opt.run}",
            f"epsilon_{int(opt.epsilon)}",
            f"maxgrad_{opt.max_grad_norm}",
            clipping_dir,
        )
    else:
        # Custom path
        result_path = opt.save_path

    return result_path


def list_available_models():
    """List all available trained models"""
    base_path = "./Snapshots/save_weights"
    models = []

    # Find all model files
    pattern = os.path.join(base_path, "**", "Inf-Net-*.pth")
    model_files = glob.glob(pattern, recursive=True)

    for model_file in model_files:
        # Extract information from path
        rel_path = os.path.relpath(model_file, base_path)
        parts = rel_path.split(os.sep)

        if len(parts) >= 3:
            model_info = {
                "path": model_file,
                "type": parts[0],
                "batch_size": (
                    parts[1].replace("batch_", "")
                    if "batch_" in parts[1]
                    else "unknown"
                ),
                "run": (
                    parts[2].replace("run_", "") if "run_" in parts[2] else "unknown"
                ),
                "epoch": os.path.basename(model_file)
                .replace("Inf-Net-", "")
                .replace(".pth", ""),
            }

            # Add additional info for DP models
            if "DP" in parts[0] and len(parts) >= 4:
                model_info["epsilon"] = parts[3].replace("epsilon_", "")

            # Add morphology info for DP_Morph models
            if "DP_Morph" in parts[0] and len(parts) >= 5:
                model_info["morph_operation"] = parts[1]
                model_info["batch_size"] = parts[2].replace("batch_", "")
                model_info["run"] = parts[3].replace("run_", "")
                model_info["epsilon"] = parts[4].replace("epsilon_", "")

            models.append(model_info)

    return models


def find_final_epoch_models():
    """Find all final epoch models for batch testing"""
    base_path = "./Snapshots/save_weights"
    final_models = []

    # Find all final epoch model files (epoch 70) for all model types
    patterns = [
        os.path.join(base_path, "**", "Inf-Net-70.pth"),
        os.path.join(base_path, "**", "UNet-70.pth"),
        os.path.join(base_path, "**", "NestedUNet-70.pth"),
    ]
    model_files = []
    for pattern in patterns:
        model_files.extend(glob.glob(pattern, recursive=True))

    for model_file in model_files:
        # Extract information from path
        rel_path = os.path.relpath(model_file, base_path)
        parts = rel_path.split(os.sep)

        model_info = {
            "path": model_file,
            "epoch": "70",
        }

        # Parse based on model type
        # Handle Inf-Net variants
        if parts[0] == "Inf-Net":
            # Structure: Inf-Net/batch_X/run_Y/Inf-Net-70.pth
            if len(parts) >= 4:
                model_info["type"] = parts[0]
                model_info["batch_size"] = parts[1].replace("batch_", "")
                model_info["run"] = parts[2].replace("run_", "")

        elif parts[0] == "Inf-Net_Morph":
            # Structure: Inf-Net_Morph/morph_op/batch_X/run_Y/Inf-Net-70.pth
            if len(parts) >= 5:
                model_info["type"] = parts[0]
                model_info["morph_operation"] = parts[1]
                model_info["batch_size"] = parts[2].replace("batch_", "")
                model_info["run"] = parts[3].replace("run_", "")

        elif parts[0] == "Inf-Net_GroupNorm":
            # Structure: Inf-Net_GroupNorm/batch_X/run_Y/Inf-Net-70.pth
            if len(parts) >= 4:
                model_info["type"] = parts[0]
                model_info["batch_size"] = parts[1].replace("batch_", "")
                model_info["run"] = parts[2].replace("run_", "")

        elif parts[0] == "Inf-Net_Morph_GroupNorm":
            # Structure: Inf-Net_Morph_GroupNorm/morph_op/batch_X/run_Y/Inf-Net-70.pth
            if len(parts) >= 5:
                model_info["type"] = parts[0]
                model_info["morph_operation"] = parts[1]
                model_info["batch_size"] = parts[2].replace("batch_", "")
                model_info["run"] = parts[3].replace("run_", "")

        elif parts[0] == "Inf-Net_DP":
            # Structure: Inf-Net_DP/batch_X/run_Y/epsilon_Z/optimizer_type/Inf-Net-70.pth
            # OR: Inf-Net_DP/batch_X/run_Y/epsilon_Z/Inf-Net-70.pth (without optimizer_type)
            if len(parts) >= 5:
                model_info["type"] = parts[0]
                model_info["batch_size"] = parts[1].replace("batch_", "")
                model_info["run"] = parts[2].replace("run_", "")
                model_info["epsilon"] = parts[3].replace("epsilon_", "")

                # Check if there's an optimizer_type subdirectory
                if len(parts) >= 6:
                    # Has optimizer_type (automatic, base, nsgd, psac)
                    model_info["optimizer_type"] = parts[4]
                # else: no optimizer_type, epsilon is at parts[3]

        elif parts[0] == "Inf-Net_DP_Morph":
            # Structure: Inf-Net_DP_Morph/morph_op/batch_X/run_Y/epsilon_Z/optimizer_type/Inf-Net-70.pth
            # OR: Inf-Net_DP_Morph/morph_op/batch_X/run_Y/epsilon_Z/Inf-Net-70.pth (without optimizer_type)
            if len(parts) >= 6:
                model_info["type"] = parts[0]
                model_info["morph_operation"] = parts[1]
                model_info["batch_size"] = parts[2].replace("batch_", "")
                model_info["run"] = parts[3].replace("run_", "")
                model_info["epsilon"] = parts[4].replace("epsilon_", "")

                # Check if there's an optimizer_type subdirectory
                if len(parts) >= 7:
                    # Has optimizer_type (automatic, base, nsgd, psac)
                    model_info["optimizer_type"] = parts[5]
                # else: no optimizer_type, epsilon is at parts[4]

        # Handle UNet variants
        elif parts[0] == "UNet_GroupNorm":
            # Structure: UNet_GroupNorm/batch_X/run_Y/UNet-70.pth
            if len(parts) >= 4:
                model_info["type"] = parts[0]
                model_info["batch_size"] = parts[1].replace("batch_", "")
                model_info["run"] = parts[2].replace("run_", "")

        elif parts[0] == "UNet_Morph_GroupNorm":
            # Structure: UNet_Morph_GroupNorm/morph_op/batch_X/run_Y/UNet-70.pth
            if len(parts) >= 5:
                model_info["type"] = parts[0]
                model_info["morph_operation"] = parts[1]
                model_info["batch_size"] = parts[2].replace("batch_", "")
                model_info["run"] = parts[3].replace("run_", "")

        elif parts[0] == "UNet_DP":
            # Structure: UNet_DP/batch_X/run_Y/epsilon_Z/optimizer_type/UNet-70.pth
            if len(parts) >= 6:
                model_info["type"] = parts[0]
                model_info["batch_size"] = parts[1].replace("batch_", "")
                model_info["run"] = parts[2].replace("run_", "")
                model_info["epsilon"] = parts[3].replace("epsilon_", "")
                model_info["optimizer_type"] = parts[4]

        elif parts[0] == "UNet_DP_Morph":
            # Structure: UNet_DP_Morph/morph_op/batch_X/run_Y/epsilon_Z/optimizer_type/UNet-70.pth
            if len(parts) >= 7:
                model_info["type"] = parts[0]
                model_info["morph_operation"] = parts[1]
                model_info["batch_size"] = parts[2].replace("batch_", "")
                model_info["run"] = parts[3].replace("run_", "")
                model_info["epsilon"] = parts[4].replace("epsilon_", "")
                model_info["optimizer_type"] = parts[5]

        # Handle NestedUNet variants
        elif parts[0] == "NestedUNet_GroupNorm":
            # Structure: NestedUNet_GroupNorm/batch_X/run_Y/NestedUNet-70.pth
            if len(parts) >= 4:
                model_info["type"] = parts[0]
                model_info["batch_size"] = parts[1].replace("batch_", "")
                model_info["run"] = parts[2].replace("run_", "")

        elif parts[0] == "NestedUNet_Morph_GroupNorm":
            # Structure: NestedUNet_Morph_GroupNorm/morph_op/batch_X/run_Y/NestedUNet-70.pth
            if len(parts) >= 5:
                model_info["type"] = parts[0]
                model_info["morph_operation"] = parts[1]
                model_info["batch_size"] = parts[2].replace("batch_", "")
                model_info["run"] = parts[3].replace("run_", "")

        elif parts[0] == "NestedUNet_DP":
            # Structure: NestedUNet_DP/batch_X/run_Y/epsilon_Z/optimizer_type/NestedUNet-70.pth
            if len(parts) >= 6:
                model_info["type"] = parts[0]
                model_info["batch_size"] = parts[1].replace("batch_", "")
                model_info["run"] = parts[2].replace("run_", "")
                model_info["epsilon"] = parts[3].replace("epsilon_", "")
                model_info["optimizer_type"] = parts[4]

        elif parts[0] == "NestedUNet_DP_Morph":
            # Structure: NestedUNet_DP_Morph/morph_op/batch_X/run_Y/epsilon_Z/optimizer_type/NestedUNet-70.pth
            if len(parts) >= 7:
                model_info["type"] = parts[0]
                model_info["morph_operation"] = parts[1]
                model_info["batch_size"] = parts[2].replace("batch_", "")
                model_info["run"] = parts[3].replace("run_", "")
                model_info["epsilon"] = parts[4].replace("epsilon_", "")
                model_info["optimizer_type"] = parts[5]

        # Only add if we successfully parsed all required fields
        if "type" in model_info and "batch_size" in model_info and "run" in model_info:
            final_models.append(model_info)

    return final_models


def run_single_test(model_info, opt):
    """Run a single test with given model info"""
    print(f"\n{'='*80}")
    print(
        f"Testing: {model_info['type']} | Batch: {model_info['batch_size']} | Run: {model_info['run']}"
    )
    if "epsilon" in model_info:
        print(f"Epsilon: {model_info['epsilon']}")
    if "morph_operation" in model_info:
        print(f"Morph Operation: {model_info['morph_operation']}")
    if "optimizer_type" in model_info:
        print(f"Optimizer Type: {model_info['optimizer_type']}")
    print(f"Model Path: {model_info['path']}")
    print(f"{'='*80}")

    # Setup device
    device = f"cuda:{opt.gpu_device}" if torch.cuda.is_available() else "cpu"

    # Load model with correct architecture
    model = load_model_for_inference(model_info["path"], model_info["type"], device)

    # Build result path matching the snapshot structure
    base_path = "./Results/Lung_infection_segmentation"
    if model_info["type"] == "Inf-Net":
        # Structure: Inf-Net/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "Inf-Net",
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
        )
    elif model_info["type"] == "Inf-Net_Morph":
        # Structure: Inf-Net_Morph/morph_op/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "Inf-Net_Morph",
            model_info["morph_operation"],
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
        )
    elif model_info["type"] == "Inf-Net_GroupNorm":
        # Structure: Inf-Net_GroupNorm/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "Inf-Net_GroupNorm",
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
        )
    elif model_info["type"] == "Inf-Net_Morph_GroupNorm":
        # Structure: Inf-Net_Morph_GroupNorm/morph_op/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "Inf-Net_Morph_GroupNorm",
            model_info["morph_operation"],
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
        )
    elif model_info["type"] == "Inf-Net_DP":
        # Structure: Inf-Net_DP/batch_X/run_Y/epsilon_Z/optimizer_type/ (if optimizer_type exists)
        # OR: Inf-Net_DP/batch_X/run_Y/epsilon_Z/ (if no optimizer_type)
        result_path_parts = [
            base_path,
            "Inf-Net_DP",
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
            f"epsilon_{int(model_info['epsilon'])}",
        ]
        if "optimizer_type" in model_info:
            result_path_parts.append(model_info["optimizer_type"])
        result_path = os.path.join(*result_path_parts)
    elif model_info["type"] == "Inf-Net_DP_Morph":
        # Structure: Inf-Net_DP_Morph/morph_op/batch_X/run_Y/epsilon_Z/optimizer_type/ (if optimizer_type exists)
        # OR: Inf-Net_DP_Morph/morph_op/batch_X/run_Y/epsilon_Z/ (if no optimizer_type)
        result_path_parts = [
            base_path,
            "Inf-Net_DP_Morph",
            model_info["morph_operation"],
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
            f"epsilon_{int(model_info['epsilon'])}",
        ]
        if "optimizer_type" in model_info:
            result_path_parts.append(model_info["optimizer_type"])
        result_path = os.path.join(*result_path_parts)
    elif model_info["type"] == "UNet_GroupNorm":
        # Structure: UNet_GroupNorm/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "UNet_GroupNorm",
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
        )
    elif model_info["type"] == "UNet_Morph_GroupNorm":
        # Structure: UNet_Morph_GroupNorm/morph_op/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "UNet_Morph_GroupNorm",
            model_info["morph_operation"],
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
        )
    elif model_info["type"] == "UNet_DP":
        # Structure: UNet_DP/batch_X/run_Y/epsilon_Z/optimizer_type/
        result_path = os.path.join(
            base_path,
            "UNet_DP",
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
            f"epsilon_{int(model_info['epsilon'])}",
            model_info["optimizer_type"],
        )
    elif model_info["type"] == "UNet_DP_Morph":
        # Structure: UNet_DP_Morph/morph_op/batch_X/run_Y/epsilon_Z/optimizer_type/
        result_path = os.path.join(
            base_path,
            "UNet_DP_Morph",
            model_info["morph_operation"],
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
            f"epsilon_{int(model_info['epsilon'])}",
            model_info["optimizer_type"],
        )
    elif model_info["type"] == "NestedUNet_GroupNorm":
        # Structure: NestedUNet_GroupNorm/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "NestedUNet_GroupNorm",
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
        )
    elif model_info["type"] == "NestedUNet_Morph_GroupNorm":
        # Structure: NestedUNet_Morph_GroupNorm/morph_op/batch_X/run_Y/
        result_path = os.path.join(
            base_path,
            "NestedUNet_Morph_GroupNorm",
            model_info["morph_operation"],
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
        )
    elif model_info["type"] == "NestedUNet_DP":
        # Structure: NestedUNet_DP/batch_X/run_Y/epsilon_Z/optimizer_type/
        result_path = os.path.join(
            base_path,
            "NestedUNet_DP",
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
            f"epsilon_{int(model_info['epsilon'])}",
            model_info["optimizer_type"],
        )
    elif model_info["type"] == "NestedUNet_DP_Morph":
        # Structure: NestedUNet_DP_Morph/morph_op/batch_X/run_Y/epsilon_Z/optimizer_type/
        result_path = os.path.join(
            base_path,
            "NestedUNet_DP_Morph",
            model_info["morph_operation"],
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
            f"epsilon_{int(model_info['epsilon'])}",
            model_info["optimizer_type"],
        )
    else:
        # Fallback for unknown types
        result_path = os.path.join(
            base_path,
            model_info["type"],
            f"batch_{model_info['batch_size']}",
            f"run_{model_info['run']}",
        )

    # Create result directory
    os.makedirs(result_path, exist_ok=True)

    # Load test data
    image_root = "{}/Imgs/".format(opt.data_path)
    test_loader = test_dataset(image_root, opt.testsize)

    print(f"Testing on {test_loader.size} images...")

    # Test loop
    for i in range(test_loader.size):
        image, name = test_loader.load_data()
        image = image.to(device)

        with torch.no_grad():
            pred = model(image)

            # Handle different model outputs
            if isinstance(pred, tuple):
                # Inf-Net outputs: (lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge)
                res = pred[3]  # Use lateral_map_2
            elif isinstance(pred, list):
                # NestedUNet with deep supervision outputs a list
                res = pred[-1]  # Use final output
            else:
                # UNet and NestedUNet (without deep supervision) output single tensor
                res = pred

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)

            save_path = os.path.join(result_path, name)
            Image.fromarray(res).save(save_path)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{test_loader.size} images")

    print(f"Test completed! Results saved to: {result_path}")
    return result_path


def inference():
    parser = argparse.ArgumentParser(
        description="Test Inf-Net models with new file structure"
    )

    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "Inf-Net",
            "Inf-Net_Morph",
            "Inf-Net_GroupNorm",
            "Inf-Net_Morph_GroupNorm",
            "Inf-Net_DP",
            "Inf-Net_DP_Morph",
            "UNet_GroupNorm",
            "UNet_Morph_GroupNorm",
            "UNet_DP",
            "UNet_DP_Morph",
            "NestedUNet_GroupNorm",
            "NestedUNet_Morph_GroupNorm",
            "NestedUNet_DP",
            "NestedUNet_DP_Morph",
            "custom",
        ],
        default="Inf-Net",
        help="Type of model to test",
    )
    parser.add_argument(
        "--batchsize", type=int, default=64, help="Batch size used during training"
    )
    parser.add_argument("--run", type=int, default=1, help="Run number")
    parser.add_argument(
        "--epoch", type=int, default=70, help="Epoch number of the model to load"
    )

    # DP-specific parameters
    parser.add_argument(
        "--epsilon",
        type=float,
        default=8.0,
        help="Epsilon (privacy budget) for DP models. Common values: 8, 200",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.2,
        help="Maximum gradient norm used during training (for DP models)",
    )
    parser.add_argument(
        "--clipping_strategy",
        type=str,
        default="base",
        choices=["base", "automatic", "psac", "nsgd"],
        help="Clipping strategy used during training (for DP models)",
    )

    # Morphology-specific parameters
    parser.add_argument(
        "--morph_operation",
        type=str,
        default="close",
        choices=["open", "close", "dilation", "both", "erosion", "none"],
        help="Morphological operation (for morphology models)",
    )
    parser.add_argument(
        "--morph_kernel_size", type=int, default=3, help="Morphological kernel size"
    )
    parser.add_argument(
        "--enable_morphology_test",
        action="store_true",
        help="Enable morphology during testing (for morphology models)",
    )

    # Custom paths (when model_type='custom')
    parser.add_argument(
        "--pth_path", type=str, default=None, help="Custom path to weights file"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Custom path to save results"
    )

    # Testing parameters
    parser.add_argument("--testsize", type=int, default=352, help="Testing size")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./Dataset/TestingSet/LungInfection-Test/",
        help="Path to test data",
    )
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU device to use")

    # Utility
    parser.add_argument(
        "--list_models", action="store_true", help="List all available trained models"
    )
    parser.add_argument(
        "--test_all_final",
        action="store_true",
        help="Test all final epoch models for different configurations",
    )
    parser.add_argument(
        "--filter_model_type",
        type=str,
        default=None,
        help="Filter models by type (e.g., 'NestedUNet_DP' to test only NestedUNet_DP models)",
    )

    opt = parser.parse_args()

    # Test all final epoch models if requested
    if opt.test_all_final:
        print("Finding all final epoch models...")
        final_models = find_final_epoch_models()

        # Filter by model type if specified
        if opt.filter_model_type:
            original_count = len(final_models)
            final_models = [
                m for m in final_models if opt.filter_model_type in m["type"]
            ]
            print(
                f"Filtered to {len(final_models)} models matching '{opt.filter_model_type}' (from {original_count} total)"
            )

        if not final_models:
            print("No final epoch models found!")
            return

        print(f"Found {len(final_models)} final epoch models:")
        for i, model in enumerate(final_models, 1):
            print(
                f"{i:3d}. {model['type']} | Batch: {model['batch_size']} | Run: {model['run']}"
            )
            if "epsilon" in model:
                print(f"     Epsilon: {model['epsilon']}")
            if "morph_operation" in model:
                print(f"     Morph Operation: {model['morph_operation']}")
            if "optimizer_type" in model:
                print(f"     Optimizer Type: {model['optimizer_type']}")

        print(f"\nStarting batch testing of {len(final_models)} models...")
        print("=" * 70)

        results_summary = []
        for i, model_info in enumerate(final_models, 1):
            try:
                print(f"\n[{i}/{len(final_models)}] Starting test...")
                result_path = run_single_test(model_info, opt)
                # Build model identifier string
                model_id = f"{model_info['type']}_batch{model_info['batch_size']}_run{model_info['run']}"
                if "epsilon" in model_info:
                    model_id += f"_eps{model_info['epsilon']}"
                if "optimizer_type" in model_info:
                    model_id += f"_{model_info['optimizer_type']}"
                if "morph_operation" in model_info:
                    model_id += f"_morph{model_info['morph_operation']}"

                results_summary.append(
                    {
                        "model": model_id,
                        "status": "SUCCESS",
                        "result_path": result_path,
                    }
                )
            except Exception as e:
                # Build model identifier string
                model_id = f"{model_info['type']}_batch{model_info['batch_size']}_run{model_info['run']}"
                if "epsilon" in model_info:
                    model_id += f"_eps{model_info['epsilon']}"
                if "optimizer_type" in model_info:
                    model_id += f"_{model_info['optimizer_type']}"
                if "morph_operation" in model_info:
                    model_id += f"_morph{model_info['morph_operation']}"

                print(f"ERROR: Failed to test {model_id}: {str(e)}")
                results_summary.append(
                    {
                        "model": model_id,
                        "status": "FAILED",
                        "error": str(e),
                    }
                )

        # Print summary
        print("\n" + "=" * 70)
        print("BATCH TESTING SUMMARY")
        print("=" * 70)
        successful = sum(1 for r in results_summary if r["status"] == "SUCCESS")
        failed = len(results_summary) - successful

        print(f"Total models tested: {len(results_summary)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if successful > 0:
            print(f"\nSuccessful tests:")
            for result in results_summary:
                if result["status"] == "SUCCESS":
                    print(f"  ✓ {result['model']} -> {result['result_path']}")

        if failed > 0:
            print(f"\nFailed tests:")
            for result in results_summary:
                if result["status"] == "FAILED":
                    print(f"  ✗ {result['model']}: {result['error']}")

        return

    # List available models if requested
    if opt.list_models:
        print("Available trained models:")
        print("-" * 80)
        models = list_available_models()
        for i, model in enumerate(models, 1):
            print(
                f"{i:3d}. {model['type']} | Batch: {model['batch_size']} | Run: {model['run']} | Epoch: {model['epoch']}"
            )
            if "epsilon" in model:
                print(f"     Epsilon: {model['epsilon']}")
            if "morph_operation" in model:
                print(f"     Morph Operation: {model['morph_operation']}")
            print(f"     Path: {model['path']}")
            print()
        return

    # Build model and result paths
    if opt.model_type == "custom":
        if not opt.pth_path:
            raise ValueError("--pth_path must be specified when model_type='custom'")
        model_path = opt.pth_path
        result_path = opt.save_path or "./Results/Lung_infection_segmentation/custom"
    else:
        model_path = build_model_path(opt)
        result_path = build_result_path(opt)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Use --list_models to see available models")
        return

    print("#" * 70)
    print(f"Start Testing ({opt.model_type})")
    print(f"Model Path: {model_path}")
    print(f"Result Path: {result_path}")
    print(f"Batch Size: {opt.batchsize}")
    print(f"Run: {opt.run}")
    print(f"Epoch: {opt.epoch}")
    if opt.model_type in ["Inf-Net_DP", "Inf-Net_DP_Morph"]:
        print(f"Epsilon: {opt.epsilon}")
    if opt.model_type in ["Inf-Net_Morph", "Inf-Net_DP_Morph"]:
        print(f"Morph Operation: {opt.morph_operation}")
        print(f"Test Morphology: {opt.enable_morphology_test}")
    print("#" * 70)

    # Setup device
    device = f"cuda:{opt.gpu_device}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model with correct architecture
    model = load_model_for_inference(model_path, opt.model_type, device)

    # Load test data
    image_root = "{}/Imgs/".format(opt.data_path)
    test_loader = test_dataset(image_root, opt.testsize)

    # Create result directory
    os.makedirs(result_path, exist_ok=True)

    print(f"Testing on {test_loader.size} images...")

    # Test loop
    for i in range(test_loader.size):
        image, name = test_loader.load_data()
        image = image.to(device)

        with torch.no_grad():
            pred = model(image)

            # Handle different model outputs
            if isinstance(pred, tuple):
                # Inf-Net outputs: (lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge)
                res = pred[3]  # Use lateral_map_2
            elif isinstance(pred, list):
                # NestedUNet with deep supervision outputs a list
                res = pred[-1]  # Use final output
            else:
                # UNet and NestedUNet (without deep supervision) output single tensor
                res = pred

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)

            save_path = os.path.join(result_path, name)
            Image.fromarray(res).save(save_path)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{test_loader.size} images")

    print(f"Test Done! Results saved to: {result_path}")


if __name__ == "__main__":
    inference()
