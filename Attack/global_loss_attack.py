import argparse
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dp_extension_codes.data_one_gpu import get_data
from dp_extension_codes.losses import CombinedLoss
from dp_extension_codes.networks import get_model

def format_epsilon_for_name(eps):
    eps = float(eps)
    if eps.is_integer():
        return str(int(eps))
    return str(eps)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def balance_two_datasets(ds1, ds2, seed=42):
    n = min(len(ds1), len(ds2))
    rng = np.random.default_rng(seed)

    def subsample(ds, n):
        if len(ds) <= n:
            return ds
        idx = rng.choice(len(ds), size=n, replace=False)
        return Subset(ds, idx.tolist())

    return subsample(ds1, n), subsample(ds2, n)


def default_batch_parser(batch):
    """
    Default parser for common dataset formats.

    Supported:
      - (x, y)
      - (x, y, extra1, ...)
      - {"x": ..., "y": ...}

    Returns:
      inputs, targets
    """
    if isinstance(batch, dict):
        x = batch["x"]
        y = batch["y"]
        return x, y

    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch tuple/list must contain at least (inputs, targets).")
        x = batch[0]
        y = batch[1]
        return x, y

    raise TypeError(f"Unsupported batch type: {type(batch)}")

# =========================================================
# Per-sample scoring
# =========================================================

@torch.no_grad()
def compute_per_sample_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    score_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, torch.device], torch.Tensor],
    device: torch.device,
    batch_parser: Callable = default_batch_parser,
) -> np.ndarray:
    """
    Compute one scalar score per sample.

    score_fn must return a tensor of shape [B], where smaller score means
    'more likely member' for a loss-based attack.
    """
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


def infer_membership_from_threshold(
    scores: np.ndarray,
    threshold: float,
    smaller_score_means_member: bool = True,
) -> np.ndarray:
    """
    Convert scores to binary predictions:
      1 = member
      0 = non-member
    """
    if smaller_score_means_member:
        return (scores <= threshold).astype(int)
    return (scores >= threshold).astype(int)


def find_best_threshold(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    smaller_score_means_member: bool = True,
    num_candidates: int = 200,
) -> Dict[str, Any]:
    """
    Pick threshold that maximizes balanced accuracy / accuracy on a calibration set.
    """
    all_scores = np.concatenate([member_scores, nonmember_scores])
    y_true = np.concatenate([
        np.ones(len(member_scores), dtype=int),
        np.zeros(len(nonmember_scores), dtype=int)
    ])

    if len(all_scores) == 0:
        raise ValueError("No scores available to choose threshold.")

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
            best = {
                "threshold": float(thr),
                "accuracy": float(acc),
            }

    return best


# =========================================================
# Attack output
# =========================================================

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


def evaluate_attack(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    threshold: float,
    smaller_score_means_member: bool = True,
) -> AttackResult:
    y_true = np.concatenate([
        np.ones(len(member_scores), dtype=int),
        np.zeros(len(nonmember_scores), dtype=int)
    ])
    all_scores = np.concatenate([member_scores, nonmember_scores])

    y_pred = infer_membership_from_threshold(
        all_scores,
        threshold=threshold,
        smaller_score_means_member=smaller_score_means_member,
    )

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # For AUC, convert to “member-likelihood score”
    if smaller_score_means_member:
        auc_scores = -all_scores
    else:
        auc_scores = all_scores

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


def save_attack_result_csv(
    result: AttackResult,
    output_csv: str,
    metadata: Optional[Dict[str, Any]] = None,
):
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

# =========================================================
# Main attack entry point
# =========================================================

def run_global_loss_attack(
    model: torch.nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    score_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, torch.device], torch.Tensor],
    device: torch.device,
    threshold: Optional[float] = None,
    smaller_score_means_member: bool = True,
    batch_parser: Callable = default_batch_parser,
    output_csv: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    auto_threshold: bool = False,
) -> AttackResult:
    """
    Generic global threshold attack based on a scalar per-sample score.

    Parameters
    ----------
    model:
        Target model.
    member_loader:
        Loader over samples that WERE used to train the model.
    nonmember_loader:
        Loader over samples that were NOT used to train the model.
    score_fn:
        Function returning per-sample scores [B].
    threshold:
        Fixed threshold. If None and auto_threshold=True, threshold is chosen
        from the same scores (quick baseline, not a strict protocol).
    smaller_score_means_member:
        For loss attacks this is usually True.
    """
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
            raise ValueError("threshold is None. Either pass threshold or set auto_threshold=True.")
        best = find_best_threshold(
            member_scores,
            nonmember_scores,
            smaller_score_means_member=smaller_score_means_member,
        )
        threshold = best["threshold"]

    result = evaluate_attack(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        threshold=threshold,
        smaller_score_means_member=smaller_score_means_member,
    )

    if output_csv is not None:
        save_attack_result_csv(result, output_csv=output_csv, metadata=metadata)

    return result

########
# handle oct based loss function
########
def segmentation_combined_score_fn_factory(criterion, use_morph=None):
    def score_fn(model, inputs, targets, device):
        if use_morph is None:
            preds = model(inputs)
        else:
            preds = model(inputs, use_morph=use_morph)

        if isinstance(preds, (list, tuple)):
            preds = preds[-1]

        batch_scores = []
        for i in range(inputs.size(0)):
            pred_i = preds[i:i+1]
            target_i = targets[i:i+1]

            if target_i.ndim == 4 and target_i.size(1) == 1:
                target_i = target_i.squeeze(1)

            loss_i = criterion(pred_i, target_i, device=device)
            batch_scores.append(loss_i)

        return torch.stack(batch_scores, dim=0)

    return score_fn

##################################################
#use a calibration set to choose threshold
#use a separate evaluation set to report attack performance
###################################################
def split_dataset_for_attack(dataset, calib_fraction=0.5, seed=42):
    n = len(dataset)
    n_calib = int(n * calib_fraction)
    n_eval = n - n_calib

    generator = torch.Generator().manual_seed(seed)
    calib_set, eval_set = random_split(dataset, [n_calib, n_eval], generator=generator)
    return calib_set, eval_set


def argument_parser():
    parser = argparse.ArgumentParser()

    # Add all arguments upfront
    parser.add_argument('--dataset', default='Duke', choices=["Duke", "UMN"])

    parser.add_argument('--batch_size', default=8 , type=int)
    parser.add_argument('--num_iterations', default=100,type=int)
    parser.add_argument('--n_classes', default=9, type=int)
    parser.add_argument('--model_name', default="unet",
                            choices=["unet", "y_net_gen", "y_net_gen_ffc", 'ReLayNet', 'UNetOrg', 'LFUNet', 'FCN8s',
                                     'NestedUNet','SimplifiedFCN8s','ConvNet'])
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--image_size', default='224', type=int)
    parser.add_argument('--g_ratio', default=0.5, type=float)
    parser.add_argument('--morphology', default=True, type=str2bool, help="morphology")
    parser.add_argument('--operation',default='erosion', type=str, help="close, open or both")
    parser.add_argument('--kernel_size', default=5, type=int, help="kernel size")
    parser.add_argument('--learnable_radius',default=False,type=str2bool)
    parser.add_argument('--use_morph',default=False,type=str2bool)
    parser.add_argument('--conditional_point',default="None",type=str,help="training,testing,validation,optional,None")
    parser.add_argument('--retinal_layer_wise', default=False, type=str2bool)
    parser.add_argument('--DPSGD', type=str2bool, default=False)
    parser.add_argument('--deep_supervision',default=True,type=str2bool)
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--epsilon', default=200, type=float)
    parser.add_argument('--excluding_3_4_layer', default=True, type=str2bool)
    parser.add_argument('--policy_type', default='None')
    parser.add_argument(
                "--clipping",
                default="flat",
                type=str,
                choices=["None","flat", "automatic", "psac", "normalized_sgd"],
                help="Gradient clipping strategy for DP-SGD (flat, automatic, psac, normalized_sgd)",
            )
    args = parser.parse_args()

    if not args.image_dir:
        if args.dataset == "UMN":
            args.image_dir = "/scicore/home/wagner0024/parsar0000/miniconda3/OCT_Sci/UMNData/"
        else:
            args.image_dir = "/scicore/home/wagner0024/parsar0000/miniconda3/OCT_Sci/DukeData/"

    return args


#### run the attack
args = argument_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = args.model_name
if model_name == "NestedUNet" and args.deep_supervision:
    model_name = "deepsuper_NestedUNet"

if args.morphology:
    final_save_dir = os.path.join("../saved_models", "morphology_models")
else:
    final_save_dir = os.path.join("../saved_models", "models")

if args.morphology:
    if args.learnable_radius:
        morph_tag = "LearnR"
    else:
        morph_tag = f"FixedR{args.kernel_size}"
else:
    morph_tag = "NoMorph"
epsilon=format_epsilon_for_name(args.epsilon) # to prevent 200.0
if args.DPSGD:
    if args.morphology:
        save_name = os.path.join(
            final_save_dir,
            f"{model_name}_{args.dataset}_DPSGD_"
            f"{args.num_iterations}_{args.batch_size}_{epsilon}_clipping_strategy{args.clipping}_"
            f"retinal_layer_wise_{args.retinal_layer_wise}_excl_three_four_{args.excluding_3_4_layer}_policy_type_{args.policy_type}_"
            f"{args.operation}_{args.kernel_size}_{morph_tag}_seed{args.seed}.pt"
        )
    else:
        save_name = os.path.join(
            final_save_dir,
            f"{model_name}_{args.dataset}_DPSGD_"
            f"{args.num_iterations}_{args.batch_size}_{epsilon}_clipping_strategy{args.clipping}_{morph_tag}_seed{args.seed}.pt"
        )
else:
    if args.morphology:
        save_name = os.path.join(
            final_save_dir,
            f"{model_name}_{args.dataset}_"
            f"{args.num_iterations}_{args.batch_size}_"
            f"retinal_layer_wise_{args.retinal_layer_wise}_excl_three_four_{args.excluding_3_4_layer}_policy_type_{args.policy_type}_"
            f"{args.operation}_{args.kernel_size}_{morph_tag}_seed{args.seed}.pt"
        )
    else:
        save_name = os.path.join(
            final_save_dir,
            f"{model_name}_{args.dataset}_"
            f"{args.num_iterations}_{args.batch_size}_{morph_tag}_seed{args.seed}.pt"
        )

print("save_name")
if args.morphology:
    model = get_model(
        args.model_name,
        ratio=args.g_ratio,
        num_classes=args.n_classes,
        morphology=args.morphology,
        operation=args.operation,
        kernel_size=args.kernel_size,
        learnable_radius=args.learnable_radius,
        use_morph=args.use_morph,
        retinal_layer_wise=args.retinal_layer_wise,
        deep_supervision=args.deep_supervision,
        policy_version=args.policy_type,
    ).to(device)
else:
    model = get_model(
        args.model_name,
        ratio=args.g_ratio,
        num_classes=args.n_classes,
        use_morph=False,
        deep_supervision=args.deep_supervision,
    ).to(device)

checkpoint = torch.load(save_name, map_location=device, weights_only=False)
state_dict = checkpoint["model_state_dict"]

new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("_module.", "")
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
model.eval()

### read data
train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_data(
    args.image_dir, args.image_size, args.batch_size
)

if not args.morphology:
    attack_use_morph = False
elif not args.conditional_point or args.conditional_point == "None":
    attack_use_morph = True
else:
    attack_use_morph = args.use_morph

criterion_attack = CombinedLoss()
score_fn = segmentation_combined_score_fn_factory(
    criterion_attack,
    use_morph=attack_use_morph
)

member_calib, member_eval = split_dataset_for_attack(train_dataset, calib_fraction=0.5, seed=args.seed)
nonmember_calib, nonmember_eval = split_dataset_for_attack(test_dataset, calib_fraction=0.5, seed=args.seed)

member_calib, nonmember_calib = balance_two_datasets(
    member_calib, nonmember_calib, seed=args.seed
)
member_eval, nonmember_eval = balance_two_datasets(
    member_eval, nonmember_eval, seed=args.seed
)

member_calib_loader = DataLoader(member_calib, batch_size=8, shuffle=False)
nonmember_calib_loader = DataLoader(nonmember_calib, batch_size=8, shuffle=False)
member_eval_loader = DataLoader(member_eval, batch_size=8, shuffle=False)
nonmember_eval_loader = DataLoader(nonmember_eval, batch_size=8, shuffle=False)

member_scores_calib = compute_per_sample_scores(model, member_calib_loader, score_fn, device)
nonmember_scores_calib = compute_per_sample_scores(model, nonmember_calib_loader, score_fn, device)

best = find_best_threshold(member_scores_calib, nonmember_scores_calib)

attack_result = run_global_loss_attack(
    model=model,
    member_loader=member_eval_loader,
    nonmember_loader=nonmember_eval_loader,
    score_fn=score_fn,
    device=device,
    threshold=best["threshold"],
    output_csv="csv_results/global_loss_attack_eval.csv",
    metadata={
        "model_name": model_name,
        "dataset": args.dataset,
        "morphology": args.morphology,
        "operation": args.operation if args.morphology else "None",
        "kernel_size": args.kernel_size if args.morphology else "None",
        "dpsgd": args.DPSGD,
        "epsilon": args.epsilon if args.DPSGD else None,
        "clipping": args.clipping if args.DPSGD else "None",
        "seed": args.seed,
        "policy_type": args.policy_type,
        "retinal_layer_wise":args.retinal_layer_wise,
        "threshold_source": "calibration_split",
    },
)

print("===== Global Loss Attack Results =====")
print(f"Threshold: {attack_result.threshold:.6f}")
print(f"Accuracy : {attack_result.accuracy:.4f}")
print(f"F1       : {attack_result.f1:.4f}")
print(f"AUC      : {attack_result.auc}")
print(f"TN={attack_result.tn}, FP={attack_result.fp}, FN={attack_result.fn}, TP={attack_result.tp}")


