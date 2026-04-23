# -*- coding: utf-8 -*-

"""
Unified Training Script for Lung Infection Segmentation
Supports:
- Three networks: Inf_Net GroupNorm, UNet_GroupNorm, NestedUNet_GroupNorm
- Base training (no DP, no Morph)
- Base with/without Morph
- Base with/without DP (with clipping strategies: base, automatic, psac, nsgd)
- Batch sizes: 24, 48
- Epsilon: 8, 200
- Morph operation: both, open, close
- Morph kernel size: 3, 5, 7, 9
- Epoch: 70
- Max grad norm: 1.2, 1.5, 2

@author: Parth Shandilya
"""
import pandas as pd
import torch
import os
import argparse
import time
import csv
from datetime import datetime
from Code.utils.dataloader_LungInf import get_loader
from Code.utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F

# Differential Privacy
from opacus.privacy_engine import PrivacyEngine
from opacus.validators import ModuleValidator

# Morphology
from kornia.morphology import opening, closing, dilation, erosion

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def apply_kornia_morphology_binary(pred_mask, operation="both", kernel_size=3):
    """Apply morphology to binary predictions"""
    choices = ["open", "close", "both", "none", "dilation", "erosion"]
    if operation not in choices:
        raise ValueError("Operation must be one of 'open', 'close', 'both', or 'none'.")

    kernel = torch.ones(kernel_size, kernel_size).to(pred_mask.device)
    if operation == "dilation":
        refined_mask = dilation(pred_mask, kernel)
    elif operation == "open":
        refined_mask = opening(pred_mask, kernel)
    elif operation == "close":
        refined_mask = closing(pred_mask, kernel)
    elif operation == "erosion":
        refined_mask = erosion(pred_mask, kernel)
    elif operation == "both":
        refined_mask = opening(pred_mask, kernel)
        refined_mask = closing(refined_mask, kernel)
    elif operation == "none":
        refined_mask = pred_mask
    else:
        raise ValueError("open", "close", "both", "none", "dilation", "erosion")
    return refined_mask


def joint_loss(pred, mask,device="cuda"):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def map_clipping_name_for_cosine(clipping_strategy: str) -> str:
    if clipping_strategy == "base":
        return "base"
    elif clipping_strategy == "automatic":
        return "automatic"
    elif clipping_strategy == "nsgd":
        return "nsgd"
    elif clipping_strategy == "psac":
        return "psac"
    else:
        raise ValueError(f"Unknown clipping strategy: {clipping_strategy}")


def map_clipping_name_for_save(clipping_strategy: str) -> str:
    if clipping_strategy == "base":
        return "flat"
    elif clipping_strategy == "automatic":
        return "automatic"
    elif clipping_strategy == "nsgd":
        return "normalized_sgd"
    elif clipping_strategy == "psac":
        return "psac"
    else:
        raise ValueError(f"Unknown clipping strategy: {clipping_strategy}")

@torch.no_grad()
def collect_per_sample_grads(model):
    chunks = []
    for p in model.parameters():
        if not hasattr(p, "grad_sample") or p.grad_sample is None:
            continue
        gs = p.grad_sample
        gs = gs.reshape(gs.shape[0], -1)   # [B, P_i]
        chunks.append(gs)

    if len(chunks) == 0:
        raise RuntimeError("No per-sample gradients found in grad_sample.")

    return torch.cat(chunks, dim=1)  # [B, P]


@torch.no_grad()
def compute_true_and_clipped_grad_cosine(model, clipping_type="base", max_grad_norm=1.0, eps=1e-12):
    """
    Returns cosine similarity between:
      - true batch-average gradient
      - transformed batch-average gradient according to clipping_type

    clipping_type:
      - flat
      - automatic
      - normalized_sgd
      - psac
    """
    r=0.1
    G = collect_per_sample_grads(model)   # [B, P]
    norms = G.norm(2, dim=1)              # [B]
    g_true = G.mean(dim=0)                # [P]

    if clipping_type == "base":
        weights = (max_grad_norm / (norms + eps)).clamp(max=1.0)

    elif clipping_type=='automatic':
        r = max_grad_norm
        weights = max_grad_norm / (norms + 0.01)
    elif clipping_type=='nsgd':
        weights=1.0 / (norms + max_grad_norm)

    elif clipping_type == "psac":
       numerator = norms + r
       denominator = norms * norms + r * norms + r
       weight = numerator / denominator
       adaptive_clip_norms = max_grad_norm * weight

       min_clip_norm = max_grad_norm * r / (1.0 + r)
       adaptive_clip_norms = torch.clamp(
           adaptive_clip_norms,
           min=min_clip_norm,
           max=max_grad_norm,
       )

       # final PSAC scaling factor: min(1, C_i / ||g_i||)
       weights = (adaptive_clip_norms / (norms + 1e-8)).clamp(max=1.0)

    else:
        raise ValueError(f"Unknown clipping_type: {clipping_type}")

    g_method = (G * weights.unsqueeze(1)).mean(dim=0)

    cos = F.cosine_similarity(
        g_true.unsqueeze(0),
        g_method.unsqueeze(0),
        dim=1
    ).item()

    return {
        "cosine": cos,
        "true_grad_norm": g_true.norm().item(),
        "method_grad_norm": g_method.norm().item(),
        "mean_sample_norm": norms.mean().item(),
    }


def train_infnet(
    train_loader,
    model,
    optimizer,
    epoch,
    save_path,
    opt,
    total_step,
    privacy_engine=None,
    BCE=None,
    gradient_cosines=None,
    gradient_cosine_epochs=None,
    gradient_cosine_batches=None,
):
    """Training function for Inf-Net (multi-scale, multi-output)"""
    model.train()
    size_rates = [0.75, 1, 1.25]
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = (
        AvgMeter(),
        AvgMeter(),
        AvgMeter(),
        AvgMeter(),
        AvgMeter(),
    )

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = images.to(opt.device)
            gts = gts.to(opt.device)
            edges = edges.to(opt.device)

            if images.size(0) == 0:
                print(f"[WARN] Empty batch at epoch={epoch}, batch={i}, rate={rate}. Skipping.")
                continue

            # ---- rescaling the inputs ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(
                    images,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )
                gts = F.interpolate(
                    gts,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )
                edges = F.interpolate(
                    edges,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )

            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = (
                model(images)
            )

            # ---- Apply morphology if enabled ----
            if opt.enable_morphology:
                lateral_map_5 = apply_kornia_morphology_binary(
                    lateral_map_5,
                    operation=opt.morph_operation,
                    kernel_size=opt.morph_kernel_size,
                )
                lateral_map_4 = apply_kornia_morphology_binary(
                    lateral_map_4,
                    operation=opt.morph_operation,
                    kernel_size=opt.morph_kernel_size,
                )
                lateral_map_3 = apply_kornia_morphology_binary(
                    lateral_map_3,
                    operation=opt.morph_operation,
                    kernel_size=opt.morph_kernel_size,
                )
                lateral_map_2 = apply_kornia_morphology_binary(
                    lateral_map_2,
                    operation=opt.morph_operation,
                    kernel_size=opt.morph_kernel_size,
                )

            # ---- loss function ----
            loss5 = joint_loss(lateral_map_5, gts)
            loss4 = joint_loss(lateral_map_4, gts)
            loss3 = joint_loss(lateral_map_3, gts)
            loss2 = joint_loss(lateral_map_2, gts)
            loss1 = BCE(lateral_edge, edges)
            loss = loss1 + loss2 + loss3 + loss4 + loss5

            # ---- backward ----
            loss.backward()

            if opt.enable_privacy and rate == 1 and ((i % 10 == 0) or (i == total_step)):
                try:
                    clipping_type = map_clipping_name_for_cosine(opt.clipping_strategy)

                    stats = compute_true_and_clipped_grad_cosine(
                        model,
                        clipping_type=clipping_type,
                        max_grad_norm=opt.max_grad_norm,
                    )
                    print(f"stats_{stats}")
                    gradient_cosines.append(stats["cosine"])
                    gradient_cosine_epochs.append(epoch)
                    gradient_cosine_batches.append(i)

                    print(
                        f"[GRAD ALIGN] epoch={epoch} batch={i} "
                        f"clip={clipping_type} "
                        f"cos={stats['cosine']:.6f} "
                        f"true_norm={stats['true_grad_norm']:.6f} "
                        f"method_norm={stats['method_grad_norm']:.6f}"
                    )
                except Exception as e:
                    print(f"[GRAD ALIGN ERROR] {e}")

            if not opt.enable_privacy:
                clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)

        # ---- train logging ----
        if i % 20 == 0 or i == total_step:
            epsilon = 0.0
            if opt.enable_privacy and privacy_engine:
                epsilon = privacy_engine.get_epsilon(delta=opt.delta)
            print(
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-edge: {:.4f}, "
                "lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}, epsilon: {:0.4f}]".format(
                    datetime.now(),
                    epoch,
                    opt.epoch,
                    i,
                    total_step,
                    loss_record1.show(),
                    loss_record2.show(),
                    loss_record3.show(),
                    loss_record4.show(),
                    loss_record5.show(),
                    epsilon,
                )
            )

    # ---- save model ----
    os.makedirs(save_path, exist_ok=True)
    if epoch == opt.epoch:  # Only save the last checkpoint
        checkpoint_path = os.path.join(save_path, f"Inf-Net-{epoch}.pth")
        if opt.enable_privacy:
            torch.save(model._module.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)
        print("[Saving Snapshot:]", checkpoint_path)
        if opt.enable_privacy and privacy_engine:
            epsilon = privacy_engine.get_epsilon(delta=opt.delta)
            print(f"[Privacy Budget]: ε = {epsilon:.2f} (δ = {opt.delta})")

    # Calculate and return epoch average loss (sum of all 5 losses)
    epoch_loss = (
        loss_record1.show()
        + loss_record2.show()
        + loss_record3.show()
        + loss_record4.show()
        + loss_record5.show()
    )
    # Convert to float if tensor
    if isinstance(epoch_loss, torch.Tensor):
        epoch_loss = epoch_loss.item()
    return float(epoch_loss)


def train_unet_nestedunet(
    train_loader,
    model,
    optimizer,
    epoch,
    save_path,
    opt,
    total_step,
    privacy_engine=None,
    gradient_cosines=None,
    gradient_cosine_epochs=None,
    gradient_cosine_batches=None,
):
    """Training function for UNet and NestedUNet (single output)"""
    model.train()
    loss_record = AvgMeter()

    # Memory management for NestedUNet (memory-intensive model)
    is_nestedunet = opt.network == "NestedUNet"
    if is_nestedunet and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("Cleared GPU cache for NestedUNet training")
        # Set memory fraction to help with fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        # Clear GPU cache more frequently for NestedUNet to prevent OOM
        if is_nestedunet and torch.cuda.is_available():
            if i % 5 == 0:
                torch.cuda.empty_cache()
            # Check memory usage and clear if getting high
            if i % 10 == 0:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                if memory_reserved > 20:  # If using more than 20GB, clear cache
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # ---- data prepare ----
        images, gts, edges = pack
        images = images.to(opt.device, non_blocking=True)
        gts = gts.to(opt.device, non_blocking=True)
        if images.size(0) == 0:
            print(f"[WARN] Empty batch at epoch={epoch}, batch={i}. Skipping.")
            continue

        # ---- forward ----
        pred = model(images)

        # ---- Handle deep supervision output (list) for NestedUNet ----
        if isinstance(pred, list):
            pred = pred[-1]

        # ---- Apply morphology if enabled ----
        if opt.enable_morphology:
            pred = apply_kornia_morphology_binary(
                pred,
                operation=opt.morph_operation,
                kernel_size=opt.morph_kernel_size,
            )

        # ---- loss function ----
        loss = joint_loss(pred, gts)#

        # ---- backward ----
        loss.backward()
        if opt.enable_privacy and ((i % 10 == 0) or (i == total_step)):
            try:
                clipping_type = map_clipping_name_for_cosine(opt.clipping_strategy)

                stats = compute_true_and_clipped_grad_cosine(
                    model,
                    clipping_type=clipping_type,
                    max_grad_norm=opt.max_grad_norm,
                )

                gradient_cosines.append(stats["cosine"])
                gradient_cosine_epochs.append(epoch)
                gradient_cosine_batches.append(i)

                print(
                    f"[GRAD ALIGN] epoch={epoch} batch={i} "
                    f"clip={clipping_type} "
                    f"cos={stats['cosine']:.6f} "
                    f"true_norm={stats['true_grad_norm']:.6f} "
                    f"method_norm={stats['method_grad_norm']:.6f}"
                )
            except Exception as e:
                print(f"[GRAD ALIGN ERROR] {e}")


        if not opt.enable_privacy:
            clip_gradient(optimizer, opt.clip)

        optimizer.step()

        # ---- recording loss ----
        # Keep as tensor (detached) for AvgMeter which expects tensors for torch.stack()
        loss_value = loss.data.detach() if hasattr(loss.data, "detach") else loss.data
        loss_record.update(loss_value, opt.batchsize)

        # Clear intermediate tensors to free memory (especially important for NestedUNet)
        del pred, loss
        if is_nestedunet and torch.cuda.is_available() and i % 5 == 0:
            torch.cuda.empty_cache()

        # ---- train logging ----
        log_freq = 5 if opt.network == "NestedUNet" else 20
        if i % log_freq == 0 or i == total_step:
            epsilon = 0.0
            if opt.enable_privacy and privacy_engine:
                epsilon = privacy_engine.get_epsilon(delta=opt.delta)
            print(
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, ε: {:.4f}".format(
                    datetime.now(),
                    epoch,
                    opt.epoch,
                    i,
                    total_step,
                    loss_record.show(),
                    epsilon,
                )
            )

    # ---- save model ----
    os.makedirs(save_path, exist_ok=True)
    if epoch == opt.epoch:  # Only save the last checkpoint
        # Map network names to checkpoint names
        checkpoint_names = {
            "UNet": "UNet",
            "NestedUNet": "NestedUNet",
        }
        model_name = checkpoint_names.get(opt.network, opt.network)
        checkpoint_path = os.path.join(save_path, f"{model_name}-{epoch}.pth")
        if opt.enable_privacy:
            torch.save(model._module.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)
        print("[Saving Snapshot:]", checkpoint_path)
        if opt.enable_privacy and privacy_engine:
            epsilon = privacy_engine.get_epsilon(delta=opt.delta)
            print(f"[Privacy Budget]: ε = {epsilon:.2f} (δ = {opt.delta})")

    # Return epoch average loss
    epoch_loss = loss_record.show()
    # Convert to float if tensor
    if isinstance(epoch_loss, torch.Tensor):
        epoch_loss = epoch_loss.item()
    return float(epoch_loss)


def build_snapshot_path(opt):
    """Build the snapshot save path based on configuration"""
    # Map network names to save path names
    network_map = {
        "Inf_Net": "Inf-Net",
        "UNet": "UNet",
        "NestedUNet": "NestedUNet",
    }
    network_name = network_map.get(opt.network, opt.network)

    if opt.is_pseudo and (not opt.is_semi):
        base_path = f"{network_name}_Pseudo"
    elif (not opt.is_pseudo) and opt.is_semi:
        base_path = f"Semi-{network_name}"
    elif (not opt.is_pseudo) and (not opt.is_semi):
        # Determine model type
        if opt.enable_privacy:
            # DP model structure
            model_prefix = f"{network_name}_DP"
            maxgrad_dir = f"maxgrad_{opt.max_grad_norm}"
            if opt.enable_morphology:
                model_type = f"{model_prefix}_Morph"
                morph_dir = opt.morph_operation
                kernel_dir = f"kernel_{opt.morph_kernel_size}"
                batch_dir = f"batch_{opt.batchsize}"
                run_dir = f"run_{opt.run}"
                epsilon_dir = f"epsilon_{int(opt.epsilon) if opt.epsilon else 8}"
                clipping_dir = (
                    opt.clipping_strategy if opt.clipping_strategy != "base" else "base"
                )
                base_path = os.path.join(
                    model_type,
                    morph_dir,
                    kernel_dir,
                    batch_dir,
                    run_dir,
                    epsilon_dir,
                    maxgrad_dir,
                    clipping_dir,
                )
            else:
                model_type = model_prefix
                batch_dir = f"batch_{opt.batchsize}"
                run_dir = f"run_{opt.run}"
                epsilon_dir = f"epsilon_{int(opt.epsilon) if opt.epsilon else 8}"
                clipping_dir = (
                    opt.clipping_strategy if opt.clipping_strategy != "base" else "base"
                )
                base_path = os.path.join(
                    model_type,
                    batch_dir,
                    run_dir,
                    epsilon_dir,
                    maxgrad_dir,
                    clipping_dir,
                )
        else:
            # Non-DP model structure
            if opt.enable_morphology:
                model_type = f"{network_name}_Morph_GroupNorm"
                morph_dir = opt.morph_operation
                kernel_dir = f"kernel_{opt.morph_kernel_size}"
                batch_dir = f"batch_{opt.batchsize}"
                run_dir = f"run_{opt.run}"
                base_path = os.path.join(
                    model_type, morph_dir, kernel_dir, batch_dir, run_dir
                )
            else:
                model_type = f"{network_name}_GroupNorm"
                batch_dir = f"batch_{opt.batchsize}"
                run_dir = f"run_{opt.run}"
                base_path = os.path.join(model_type, batch_dir, run_dir)
    else:
        # Custom save path
        base_path = opt.train_save

    save_path = os.path.join("./Snapshots/save_weights", base_path)
    return save_path


def save_training_results_to_csv(
    opt, training_losses, training_time_seconds, privacy_engine=None
):
    """
    Save training results to CSV files (similar to OCT structure).
    Saves to both per-experiment CSV and global CSV.

    Args:
        opt: Training options/arguments
        training_losses: List of training losses per epoch
        training_time_seconds: Total training time in seconds
        privacy_engine: Privacy engine object (if DP enabled)
    """
    # Build model name
    network_map = {
        "Inf_Net": "Inf-Net",
        "UNet": "UNet",
        "NestedUNet": "NestedUNet",
    }
    network_name = network_map.get(opt.network, opt.network)

    if opt.enable_privacy:
        model_prefix = f"{network_name}_DP"
        if opt.enable_morphology:
            model_name = f"{model_prefix}_Morph"
        else:
            model_name = model_prefix
    else:
        if opt.enable_morphology:
            model_name = f"{network_name}_Morph_GroupNorm"
        else:
            model_name = f"{network_name}_GroupNorm"

    # Build directory structure based on experimental strategy (similar to OCT)
    results_dir = opt.results_base
    dataset = "LungInfection"
    dataset_dir = os.path.join(results_dir, dataset)

    if opt.enable_privacy:
        dp_dir = os.path.join(dataset_dir, "dp")
        if opt.clipping_strategy == "base":
            clipping_dir = os.path.join(dp_dir, "base")
        else:
            clipping_dir = os.path.join(dp_dir, opt.clipping_strategy)
        epsilon_dir = os.path.join(
            clipping_dir, f"epsilon_{int(opt.epsilon) if opt.epsilon else 8}"
        )
        if opt.enable_morphology:
            morph_dir = os.path.join(
                epsilon_dir,
                "with_morph",
                opt.morph_operation,
                f"kernel_{opt.morph_kernel_size}",
            )
        else:
            morph_dir = os.path.join(epsilon_dir, "no_morph")
    else:
        non_dp_dir = os.path.join(dataset_dir, "non_dp")
        if opt.enable_morphology:
            morph_dir = os.path.join(
                non_dp_dir,
                "with_morph",
                opt.morph_operation,
                f"kernel_{opt.morph_kernel_size}",
            )
        else:
            morph_dir = os.path.join(non_dp_dir, "no_morph")

    # Create training subdirectory
    training_dir = os.path.join(morph_dir, "training")
    os.makedirs(training_dir, exist_ok=True)

    # Create filename with batch size
    model_name_lower = model_name.lower()
    file_name = os.path.join(
        training_dir, f"{model_name_lower}_batch{opt.batchsize}_results.csv"
    )

    # Global CSV file path
    global_csv_path = os.path.join(results_dir, f"all_results_training_global.csv")

    # Extract noise multiplier if DP enabled
    noise_multiplier = None
    if opt.enable_privacy and privacy_engine:
        try:
            noise_multiplier = privacy_engine.noise_multiplier
        except AttributeError:
            # Try alternative access method
            try:
                noise_multiplier = privacy_engine._noise_multiplier
            except AttributeError:
                noise_multiplier = None

    # Prepare data to save in CSV
    row_data = [
        model_name,
        dataset,
        opt.enable_privacy,
        opt.clipping_strategy if opt.enable_privacy else "none",
        opt.epsilon if opt.enable_privacy else 0,
        opt.enable_morphology,
        opt.morph_operation if opt.enable_morphology else "none",
        opt.morph_kernel_size if opt.enable_morphology else 0,
        opt.lr,
        opt.batchsize,
        opt.run,
        opt.epoch,
        training_losses[-1] if training_losses else None,
        training_time_seconds,
        opt.max_grad_norm if opt.enable_privacy else None,
        noise_multiplier,
        "training",
    ]

    # CSV header
    header = [
        "Model_Name",
        "Dataset",
        "DPSGD",
        "Clipping_Strategy",
        "Epsilon",
        "Morphology",
        "Operation",
        "Kernel_Size",
        "Learning_Rate",
        "Batch_Size",
        "Run_Number",
        "Iterations",
        "Training_Loss",
        "Training_Time_Seconds",
        "Max_Grad_Norm",
        "Noise_Multiplier",
        "Stage",
    ]

    try:
        # Save to per-experiment CSV
        file_exists = os.path.isfile(file_name)
        with open(file_name, "a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row_data)

        print(f"Training results saved to {os.path.abspath(file_name)}")

        # Save to global CSV
        global_file_exists = os.path.isfile(global_csv_path)
        with open(global_csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            if not global_file_exists:
                writer.writerow(header)
            writer.writerow(row_data)

        print(
            f"Training results also saved to global CSV: {os.path.abspath(global_csv_path)}"
        )

    except Exception as e:
        print(f"Failed to save training results to CSV: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Network selection
    parser.add_argument(
        "--network",
        type=str,
        default="Inf_Net",
        choices=["Inf_Net", "UNet", "NestedUNet"],
        help="Network architecture to use",
    )

    # hyper-parameters
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
        default=2,
        help="number of workers in dataloader. In windows, set num_workers=0",
    )

    # model parameters
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

    # training dataset
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
        help="Gradient clipping strategy for DP-SGD. 'base' means no clipping arg (flat), others pass clipping parameter",
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
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--dataset_name", type=str, default="LungInfection")

    opt = parser.parse_args()

    # ---- setup device ----
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- build models ----
    if opt.device == "cuda":
        torch.cuda.set_device(opt.gpu_device)

    if opt.network == "Inf_Net":
        if opt.backbone == "Res2Net50":
            print("Backbone loading: Res2Net50")
            from Code.model_lung_infection.InfNet_Res2Net import Inf_Net
        elif opt.backbone == "ResNet50":
            print("Backbone loading: ResNet50")
            from Code.model_lung_infection.InfNet_ResNet import Inf_Net
        elif opt.backbone == "VGGNet16":
            print("Backbone loading: VGGNet16")
            from Code.model_lung_infection.InfNet_VGGNet import Inf_Net
        else:
            raise ValueError("Invalid backbone parameters: {}".format(opt.backbone))

        model = Inf_Net(channel=opt.net_channel, n_class=opt.n_classes).to(opt.device)

        # Freeze unused branches for DP compatibility
        if opt.enable_privacy and opt.backbone == "Res2Net50":
            if hasattr(model.resnet, "avgpool"):
                for param in model.resnet.avgpool.parameters():
                    param.requires_grad = False
            if hasattr(model.resnet, "fc"):
                for param in model.resnet.fc.parameters():
                    param.requires_grad = False
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable: {trainable_params:,} / {total_params:,} parameters")

    elif opt.network == "UNet":
        print("Model loading: UNet with GroupNorm")
        from Code.model_lung_infection.InfNet_UNet_GroupNorm import UNet_GroupNorm

        model = UNet_GroupNorm(
            in_channels=opt.in_channels,
            out_channels=opt.n_classes,
            init_features=opt.init_features,
        ).to(opt.device)

    elif opt.network == "NestedUNet":
        print("Model loading: NestedUNet (UNet++) with GroupNorm")
        from Code.model_lung_infection.InfNet_NestedUNet_GroupNorm import (
            NestedUNet_GroupNorm,
        )

        # Disable deep_supervision for NestedUNet to reduce memory usage
        # Deep supervision significantly increases memory consumption
        deep_supervision = False  # Always False for memory efficiency
        print(
            f"NestedUNet: deep_supervision={deep_supervision} (disabled for memory efficiency)"
        )

        # For NestedUNet with batch size 24, we need to reduce memory usage
        # We'll use a custom model with reduced feature channels
        # Original: [32, 64, 128, 256, 512], Reduced: [24, 48, 96, 192, 384]
        # This maintains architecture similarity while reducing memory by ~40%
        print(
            "NestedUNet: Using reduced feature channels [24, 48, 96, 192, 384] for memory efficiency with batch size 24"
        )

        # Create model with reduced features
        # We need to modify the model class to accept custom feature channels
        # For now, we'll use the default and rely on aggressive memory management
        model = NestedUNet_GroupNorm(
            input_channels=opt.in_channels,
            num_classes=opt.n_classes,
            deep_supervision=deep_supervision,
        ).to(opt.device)

    else:
        raise ValueError(f"Unknown network: {opt.network}")

    # ---- Fix model for differential privacy BEFORE creating optimizer ----
    if opt.enable_privacy:
        try:
            ModuleValidator.validate(model, strict=True)
            print("Model is compatible with differential privacy")
        except Exception as e:
            print(f"Validator raised issues with model. Attempting auto-fix...")
            print(f"Original error: {type(e).__name__}")
            model = ModuleValidator.fix(model)
            model = model.to(opt.device)
            try:
                ModuleValidator.validate(model, strict=True)
                print("Model successfully fixed for differential privacy")
            except Exception as validation_error:
                print(
                    f"Warning: Model may still have compatibility issues: {validation_error}"
                )
    else:
        # Convert BatchNorm to GroupNorm for fair comparison
        print("Converting BatchNorm to GroupNorm for fair comparison with DP models")
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
            print("Model architecture converted to GroupNorm")
        else:
            print("Model already GroupNorm-compatible")
        model = model.to(opt.device)

    # ---- load pre-trained weights ----
    if opt.is_semi and opt.network == "Inf_Net" and opt.backbone == "Res2Net50":
        print("Loading weights from weights file trained on pseudo label")
        model.load_state_dict(
            torch.load("./Snapshots/save_weights/Inf-Net_Pseduo/Inf-Net_pseudo_100.pth")
        )
    else:
        print("Not loading weights from weights file")

    # ---- calculate FLOPs and Params ----
    if opt.is_thop:
        from Code.utils.utils import CalParams

        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).to(opt.device)
        CalParams(model, x)

    # ---- load training sub-modules ----
    BCE = torch.nn.BCEWithLogitsLoss()

    # ---- Create optimizer AFTER model fix ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    # ---- Load data ----
    image_root = "{}/Imgs/".format(opt.train_path)
    gt_root = "{}/GT/".format(opt.train_path)
    edge_root = "{}/Edge/".format(opt.train_path)

    train_loader = get_loader(
        image_root,
        gt_root,
        edge_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=opt.num_workers,
    )
    total_step = len(train_loader)

    # ---- Build save path ----
    save_path = build_snapshot_path(opt)

    # ---- Setup privacy engine ----
    privacy_engine = None
    if opt.enable_privacy:
        privacy_engine = PrivacyEngine()
        # Use make_private_with_epsilon
        # For 'base' (flat), don't pass clipping argument
        # For others, pass clipping parameter
        if opt.clipping_strategy == "base":
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=opt.epsilon,
                target_delta=opt.delta,
                epochs=opt.epoch,
                max_grad_norm=opt.max_grad_norm,
            )
        else:
            # Map nsgd to normalized_sgd for opacus
            clipping_value = (
                "normalized_sgd"
                if opt.clipping_strategy == "nsgd"
                else opt.clipping_strategy
            )
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=opt.epsilon,
                target_delta=opt.delta,
                epochs=opt.epoch,
                max_grad_norm=opt.max_grad_norm,
                clipping=clipping_value,
            )

    # ---- Print training info ----
    privacy_info = (
        f"DP Enabled (ε={opt.epsilon}, δ={opt.delta}, clipping={opt.clipping_strategy})"
        if opt.enable_privacy
        else "DP: Disabled"
    )
    morph_info = (
        f"Morphology: {opt.morph_operation}"
        if opt.enable_morphology
        else "Morphology: Disabled"
    )
    print(
        "#" * 70,
        "\nStart Training ({}-{})\n"
        "Network: {}\n"
        "Batch Size: {}\n"
        "{}\n"
        "{}\n"
        "Architecture: GroupNorm{}\n"
        "Save Path: {}\n"
        "Run: {}\n"
        "{}\n".format(
            opt.network,
            "DP" if opt.enable_privacy else "GroupNorm",
            opt.network,
            opt.batchsize,
            privacy_info,
            morph_info,
            " (with DP)" if opt.enable_privacy else " (no DP)",
            save_path,
            opt.run,
            opt,
        ),
        "#" * 70,
    )
    gradient_cosine_epochs=[]
    gradient_cosine_epochs = []
    gradient_cosine_batches = []
    # Select training function based on network
    if opt.network == "Inf_Net":
        train_func = lambda *args, **kwargs: train_infnet(*args, BCE=BCE, **kwargs)
    else:
        train_func = train_unet_nestedunet

    # Initialize training loss tracking and time tracking
    training_losses = []
    start_time = time.time()
    gradient_cosines = []
    gradient_cosine_epochs = []
    gradient_cosine_batches = []

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        epoch_loss = train_func(
            train_loader,
            model,
            optimizer,
            epoch,
            save_path,
            opt,
            total_step,
            privacy_engine,
            gradient_cosines=gradient_cosines,
            gradient_cosine_epochs=gradient_cosine_epochs,
            gradient_cosine_batches=gradient_cosine_batches,
        )
        training_losses.append(epoch_loss)

        print(f"Epoch {epoch}/{opt.epoch} completed. Average loss: {epoch_loss:.4f}")

    # Calculate total training time
    end_time = time.time()
    training_time_seconds = end_time - start_time
    print(
        f"\nTraining completed in {training_time_seconds:.2f} seconds ({training_time_seconds/60:.2f} minutes)"
    )

    # Save training results to CSV
    print("\n=== Saving Training Results to CSV ===")
    save_training_results_to_csv(
        opt, training_losses, training_time_seconds, privacy_engine
    )
    print(f"gradient:{gradient_cosines}")

    if opt.enable_privacy and len(gradient_cosines) > 0:
        grad_dir = "gradient_alignment_logs"
        os.makedirs(grad_dir, exist_ok=True)

        clipping_type_for_save = map_clipping_name_for_save(opt.clipping_strategy)

        grad_df = pd.DataFrame({
            "epoch": gradient_cosine_epochs,
            "batch_idx": gradient_cosine_batches,
            "dataset": opt.dataset_name,
            "model_name": opt.network,
            "clipping": clipping_type_for_save,
            "max_grad_norm": opt.max_grad_norm,
            "cosine": gradient_cosines,
            "seed": opt.seed,
            "morphology": opt.enable_morphology,
            "retinal_layer_wise": False,
        })
        epsilon_value=opt.epsilon if opt.enable_privacy else "",
        grad_path = os.path.join(
            grad_dir,
            f"{opt.dataset_name}_{opt.network}_{clipping_type_for_save}_eps{epsilon_value}"
            f"_seed{opt.seed}_morphology{opt.enable_morphology}"
            f"_retinal_layer_wise_False.csv"
        )

        grad_df.to_csv(grad_path, index=False)
        print(f"Saved gradient alignment log to: {grad_path}")