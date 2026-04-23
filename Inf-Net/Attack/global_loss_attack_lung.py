import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from opacus.validators import ModuleValidator
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Code.utils.dataloader_LungInf import get_loader
from Code.utils.dataloader_LungInf import COVIDDataset

# =========================================================
# Helpers
# =========================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def format_epsilon_for_name(eps):
    eps = float(eps)
    if eps.is_integer():
        return str(int(eps))
    return str(eps)


def balance_two_datasets(ds1, ds2, seed=42):
    n = min(len(ds1), len(ds2))
    rng = np.random.default_rng(seed)

    def subsample(ds, n_target):
        if len(ds) <= n_target:
            return ds
        idx = rng.choice(len(ds), size=n_target, replace=False)
        return Subset(ds, idx.tolist())

    return subsample(ds1, n), subsample(ds2, n)


def split_dataset_for_attack(dataset, calib_fraction=0.5, seed=42):
    n = len(dataset)
    n_calib = int(n * calib_fraction)
    n_eval = n - n_calib
    generator = torch.Generator().manual_seed(seed)
    calib_set, eval_set = random_split(dataset, [n_calib, n_eval], generator=generator)
    return calib_set, eval_set


# =========================================================
# Loss used in the original lung training
# =========================================================
def joint_loss(pred, mask):
    """
    pred: [B,1,H,W] logits
    mask: [B,1,H,W] float target in {0,1}
    """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def per_sample_joint_loss(pred, mask):
    """
    pred: [B,1,H,W] logits
    mask: [B,1,H,W] float target
    returns: [B]
    """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    loss = wbce + wiou   # [B,1] or [B]
    return loss.view(loss.shape[0])


# =========================================================
# Morphology
# =========================================================
from kornia.morphology import opening, closing, dilation, erosion

def apply_kornia_morphology_binary(pred_mask, operation="both", kernel_size=3):
    kernel = torch.ones(kernel_size, kernel_size, device=pred_mask.device)

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
        raise ValueError(f"Unknown morphology operation: {operation}")
    return refined_mask


# =========================================================
# Attack core
# =========================================================
def default_batch_parser(batch):
    """
    Lung loader returns (images, gts, edges)
    We only need images and gts.
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch must contain at least (images, masks)")
        x = batch[0]
        y = batch[1]
        return x, y

    if isinstance(batch, dict):
        return batch["x"], batch["y"]

    raise TypeError(f"Unsupported batch type: {type(batch)}")


@torch.no_grad()
def compute_per_sample_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    score_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, torch.device], torch.Tensor],
    device: torch.device,
    batch_parser: Callable = default_batch_parser,
) -> np.ndarray:
    model.eval()
    all_scores = []

    for batch in dataloader:
        inputs, targets = batch_parser(batch)
        inputs = inputs.to(device)
        targets = targets.to(device)

        scores = score_fn(model, inputs, targets, device)

        if not isinstance(scores, torch.Tensor):
            raise TypeError("score_fn must return a torch.Tensor.")
        if scores.ndim != 1:
            raise ValueError(f"score_fn must return shape [B], got {scores.shape}.")

        all_scores.append(scores.detach().cpu().numpy())

    if len(all_scores) == 0:
        return np.array([], dtype=np.float32)

    return np.concatenate(all_scores, axis=0)


def infer_membership_from_threshold(scores, threshold, smaller_score_means_member=True):
    if smaller_score_means_member:
        return (scores <= threshold).astype(int)
    return (scores >= threshold).astype(int)


def find_best_threshold(member_scores, nonmember_scores, smaller_score_means_member=True, num_candidates=200):
    all_scores = np.concatenate([member_scores, nonmember_scores])
    y_true = np.concatenate([
        np.ones(len(member_scores), dtype=int),
        np.zeros(len(nonmember_scores), dtype=int)
    ])

    if len(all_scores) == 0:
        raise ValueError("No scores available.")

    candidates = np.linspace(all_scores.min(), all_scores.max(), num_candidates)

    best = None
    best_acc = -1.0

    for thr in candidates:
        y_pred = infer_membership_from_threshold(
            all_scores, thr, smaller_score_means_member=smaller_score_means_member
        )
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best = {"threshold": float(thr), "accuracy": float(acc)}

    return best


@dataclass
class AttackResult:
    threshold: float
    accuracy: float
    f1: float
    auc: Optional[float]
    tn: int
    fp: int
    fn: int
    tp: int
    member_scores: np.ndarray
    nonmember_scores: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


def evaluate_attack(member_scores, nonmember_scores, threshold, smaller_score_means_member=True):
    y_true = np.concatenate([
        np.ones(len(member_scores), dtype=int),
        np.zeros(len(nonmember_scores), dtype=int)
    ])
    all_scores = np.concatenate([member_scores, nonmember_scores])

    y_pred = infer_membership_from_threshold(
        all_scores, threshold, smaller_score_means_member=smaller_score_means_member
    )

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    auc_scores = -all_scores if smaller_score_means_member else all_scores
    try:
        auc = roc_auc_score(y_true, auc_scores)
    except Exception:
        auc = None

    return AttackResult(
        threshold=float(threshold),
        accuracy=float(acc),
        f1=float(f1),
        auc=None if auc is None else float(auc),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        y_true=y_true,
        y_pred=y_pred,
    )


def save_attack_result_csv(result: AttackResult, output_csv: str, metadata: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    row = {
        "threshold": result.threshold,
        "accuracy": result.accuracy,
        "f1": result.f1,
        "auc": result.auc,
        "tn": result.tn,
        "fp": result.fp,
        "fn": result.fn,
        "tp": result.tp,
        "num_members": len(result.member_scores),
        "num_nonmembers": len(result.nonmember_scores),
        "member_scores": [result.member_scores.tolist()],
        "nonmember_scores": [result.nonmember_scores.tolist()],
        "y_true": [result.y_true.tolist()],
        "y_pred": [result.y_pred.tolist()],
    }

    if metadata:
        row.update(metadata)

    df = pd.DataFrame([row])
    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(output_csv, index=False)


def run_global_loss_attack(
    model,
    member_loader,
    nonmember_loader,
    score_fn,
    device,
    threshold=None,
    smaller_score_means_member=True,
    batch_parser=default_batch_parser,
    output_csv=None,
    metadata=None,
    auto_threshold=False,
):
    member_scores = compute_per_sample_scores(
        model=model,
        dataloader=member_loader,
        score_fn=score_fn,
        device=device,
        batch_parser=batch_parser,
    )
    nonmember_scores = compute_per_sample_scores(
        model=model,
        dataloader=nonmember_loader,
        score_fn=score_fn,
        device=device,
        batch_parser=batch_parser,
    )

    if threshold is None:
        if not auto_threshold:
            raise ValueError("threshold is None. Pass threshold or set auto_threshold=True.")
        best = find_best_threshold(
            member_scores, nonmember_scores,
            smaller_score_means_member=smaller_score_means_member
        )
        threshold = best["threshold"]

    result = evaluate_attack(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        threshold=threshold,
        smaller_score_means_member=smaller_score_means_member,
    )

    if output_csv is not None:
        save_attack_result_csv(result, output_csv, metadata=metadata)

    return result


# =========================================================
# Build/load lung model
# =========================================================
def build_lung_model(opt):
    if opt.network == "Inf_Net":
        if opt.backbone == "Res2Net50":
            from Code.model_lung_infection.InfNet_Res2Net import Inf_Net
        elif opt.backbone == "ResNet50":
            from Code.model_lung_infection.InfNet_ResNet import Inf_Net
        elif opt.backbone == "VGGNet16":
            from Code.model_lung_infection.InfNet_VGGNet import Inf_Net
        else:
            raise ValueError(f"Invalid backbone: {opt.backbone}")

        model = Inf_Net(channel=opt.net_channel, n_class=opt.n_classes).to(opt.device)

    elif opt.network == "UNet":
        from Code.model_lung_infection.InfNet_UNet_GroupNorm import UNet_GroupNorm

        model = UNet_GroupNorm(
            in_channels=opt.in_channels,
            out_channels=opt.n_classes,
            init_features=opt.init_features,
        ).to(opt.device)

    elif opt.network == "NestedUNet":
        from Code.model_lung_infection.InfNet_NestedUNet_GroupNorm import NestedUNet_GroupNorm

        model = NestedUNet_GroupNorm(
            input_channels=opt.in_channels,
            num_classes=opt.n_classes,
            deep_supervision=False,
        ).to(opt.device)
    else:
        raise ValueError(f"Unknown network: {opt.network}")

    return model


def build_snapshot_path(opt):
    network_map = {
        "Inf_Net": "Inf-Net",
        "UNet": "UNet",
        "NestedUNet": "NestedUNet",
    }
    network_name = network_map.get(opt.network, opt.network)

    if opt.enable_privacy:
        model_prefix = f"{network_name}_DP"
        maxgrad_dir = f"maxgrad_{opt.max_grad_norm}"

        if opt.enable_morphology:
            model_type = f"{model_prefix}_Morph"
            morph_dir = opt.morph_operation
            kernel_dir = f"kernel_{opt.morph_kernel_size}"
            batch_dir = f"batch_{opt.batchsize}"
            run_dir = f"run_{opt.run}"
            epsilon_dir = f"epsilon_{int(opt.epsilon)}"
            clipping_dir = opt.clipping_strategy if opt.clipping_strategy != "base" else "base"
            base_path = os.path.join(
                model_type, morph_dir, kernel_dir, batch_dir, run_dir,
                epsilon_dir, maxgrad_dir, clipping_dir
            )
        else:
            model_type = model_prefix
            batch_dir = f"batch_{opt.batchsize}"
            run_dir = f"run_{opt.run}"
            epsilon_dir = f"epsilon_{int(opt.epsilon)}"
            clipping_dir = opt.clipping_strategy if opt.clipping_strategy != "base" else "base"
            base_path = os.path.join(
                model_type, batch_dir, run_dir, epsilon_dir, maxgrad_dir, clipping_dir
            )
    else:
        if opt.enable_morphology:
            model_type = f"{network_name}_Morph_GroupNorm"
            morph_dir = opt.morph_operation
            kernel_dir = f"kernel_{opt.morph_kernel_size}"
            batch_dir = f"batch_{opt.batchsize}"
            run_dir = f"run_{opt.run}"
            base_path = os.path.join(model_type, morph_dir, kernel_dir, batch_dir, run_dir)
        else:
            model_type = f"{network_name}_GroupNorm"
            batch_dir = f"batch_{opt.batchsize}"
            run_dir = f"run_{opt.run}"
            base_path = os.path.join(model_type, batch_dir, run_dir)

    return os.path.join("../Snapshots/save_weights", base_path)

def build_checkpoint_file(opt):
    network_map = {
        "Inf_Net": "Inf-Net",
        "UNet": "UNet",
        "NestedUNet": "NestedUNet",
    }
    network_name = network_map.get(opt.network, opt.network)
    snapshot_dir = build_snapshot_path(opt)
    return os.path.join(snapshot_dir, f"{network_name}-{opt.epoch}.pth")

# =========================================================
# Score function for lung models
# =========================================================
def lung_score_fn_factory(opt):
    def score_fn(model, inputs, targets, device):
        outputs = model(inputs)

        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        if opt.network == "Inf_Net":
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = outputs

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

            loss5 = per_sample_joint_loss(lateral_map_5, targets)
            loss4 = per_sample_joint_loss(lateral_map_4, targets)
            loss3 = per_sample_joint_loss(lateral_map_3, targets)
            loss2 = per_sample_joint_loss(lateral_map_2, targets)

            return loss5 + loss4 + loss3 + loss2

        else:
            preds = outputs
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]

            if opt.enable_morphology:
                preds = apply_kornia_morphology_binary(
                    preds,
                    operation=opt.morph_operation,
                    kernel_size=opt.morph_kernel_size,
                )

            return per_sample_joint_loss(preds, targets)

    return score_fn

# =========================================================
# Args
# =========================================================
def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--network", type=str, default="Inf_Net",
                        choices=["Inf_Net", "UNet", "NestedUNet"])
    parser.add_argument("--epoch", type=int, default=70)
    parser.add_argument("--batchsize", type=int, default=24)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--run", type=int, default=1)

    parser.add_argument("--enable_privacy", action="store_true")
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.2)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clipping_strategy", type=str, default="base",
                        choices=["base", "automatic", "psac", "nsgd"])

    parser.add_argument("--enable_morphology", action="store_true")
    parser.add_argument("--morph_operation", type=str, default="both",
                        choices=["open", "close", "dilation", "erosion", "both"])
    parser.add_argument("--morph_kernel_size", type=int, default=3)

    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="LungInfection")

    parser.add_argument("--train_path", type=str,
                        default="../Dataset/TrainingSet/LungInfection-Train/Doctor-label")
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="../Dataset/TestingSet/LungInfection-Test/",
    )

    parser.add_argument("--backbone", type=str, default="Res2Net50")
    parser.add_argument("--net_channel", type=int, default=32)
    parser.add_argument("--init_features", type=int, default=32)
    parser.add_argument("--n_classes", type=int, default=1)
    parser.add_argument("--in_channels", type=int, default=3)

    parser.add_argument("--calib_fraction", type=float, default=0.5)
    parser.add_argument("--attack_batch_size", type=int, default=8)
    parser.add_argument("--output_csv", type=str, default="results/global_loss_attack_lung.csv")

    opt = parser.parse_args()
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    return opt


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    opt = argument_parser()

    if opt.device == "cuda":
        torch.cuda.set_device(opt.gpu_device)

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    print("Building model...")
    model = build_lung_model(opt)
    try:
        ModuleValidator.validate(model, strict=True)
    except Exception:
        model = ModuleValidator.fix(model)

    model = model.to(opt.device)

    checkpoint_path = build_checkpoint_file(opt)
    print("Loading checkpoint from:", checkpoint_path)

    state_dict = torch.load(checkpoint_path, map_location=opt.device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    print("Loading data...")

    # members = train set


    member_dataset = COVIDDataset(
        image_root=f"{opt.train_path}/Imgs/",
        gt_root=f"{opt.train_path}/GT/",
        edge_root=f"{opt.train_path}/Edge/",
        trainsize=opt.trainsize,
    )

    nonmember_dataset = COVIDDataset(
                            image_root=f"{opt.test_data_path}/Imgs/",
                            gt_root=f"{opt.test_data_path}/GT/",
                            edge_root="",
                            trainsize=opt.trainsize,
                        )

    member_calib, member_eval = split_dataset_for_attack(
        member_dataset, calib_fraction=opt.calib_fraction, seed=opt.seed
    )
    nonmember_calib, nonmember_eval = split_dataset_for_attack(
        nonmember_dataset, calib_fraction=opt.calib_fraction, seed=opt.seed
    )

    member_calib, nonmember_calib = balance_two_datasets(member_calib, nonmember_calib, seed=opt.seed)
    member_eval, nonmember_eval = balance_two_datasets(member_eval, nonmember_eval, seed=opt.seed)

    member_calib_loader = DataLoader(member_calib, batch_size=opt.attack_batch_size, shuffle=False)
    nonmember_calib_loader = DataLoader(nonmember_calib, batch_size=opt.attack_batch_size, shuffle=False)
    member_eval_loader = DataLoader(member_eval, batch_size=opt.attack_batch_size, shuffle=False)
    nonmember_eval_loader = DataLoader(nonmember_eval, batch_size=opt.attack_batch_size, shuffle=False)

    score_fn = lung_score_fn_factory(opt)

    print("Computing calibration scores...")
    member_scores_calib = compute_per_sample_scores(model, member_calib_loader, score_fn, opt.device)
    nonmember_scores_calib = compute_per_sample_scores(model, nonmember_calib_loader, score_fn, opt.device)

    best = find_best_threshold(member_scores_calib, nonmember_scores_calib)

    print("Running evaluation attack...")
    attack_result = run_global_loss_attack(
        model=model,
        member_loader=member_eval_loader,
        nonmember_loader=nonmember_eval_loader,
        score_fn=score_fn,
        device=opt.device,
        threshold=best["threshold"],
        output_csv=opt.output_csv,
        metadata={
            "network": opt.network,
            "dataset": opt.dataset_name,
            "enable_privacy": opt.enable_privacy,
            "epsilon": opt.epsilon if opt.enable_privacy else None,
            "clipping_strategy": opt.clipping_strategy if opt.enable_privacy else "none",
            "enable_morphology": opt.enable_morphology,
            "morph_operation": opt.morph_operation if opt.enable_morphology else "none",
            "morph_kernel_size": opt.morph_kernel_size if opt.enable_morphology else None,
            "batchsize": opt.batchsize,
            "run": opt.run,
            "seed": opt.seed,
            "threshold_source": "calibration_split",
        },
    )

    print("===== Global Loss Attack Results =====")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Threshold: {attack_result.threshold:.6f}")
    print(f"Accuracy : {attack_result.accuracy:.4f}")
    print(f"F1       : {attack_result.f1:.4f}")
    print(f"AUC      : {attack_result.auc}")
    print(f"TN={attack_result.tn}, FP={attack_result.fp}, FN={attack_result.fn}, TP={attack_result.tp}")