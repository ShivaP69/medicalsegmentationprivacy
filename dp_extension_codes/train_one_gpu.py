import random
import argparse
import torch
import tqdm
import time
import csv
from data_one_gpu import get_data
from losses import CombinedLoss, FocalFrequencyLoss
from networks import get_model
from utils import per_class_dice
from utils import MAE_New
from utils import compute_dice
from utils import hd95_multiclass
from utils import compute_pa
from opacus import PrivacyEngine
from kornia.morphology import opening, closing,dilation,erosion
import os
import numpy as np
import pandas as pd
#from fastDP import PrivacyEngine
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn as nn
torch.cuda.empty_cache()
print("Current Directory:", os.getcwd())
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
import torch.nn.functional as F

# to get a layer
def get_module_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    modules_list = list(model.named_modules())
    modules_dict = dict(modules_list)

    # 1) exact match
    if layer_name in modules_dict:
        return modules_dict[layer_name]

    # If it's a digit, interpret as index into named_modules()
    if layer_name.isdigit():
        idx = int(layer_name)
        if idx < 0 or idx >= len(modules_list):
            raise IndexError(
                f"Index {idx} out of range for named_modules (len={len(modules_list)})."
            )
        name, module = modules_list[idx]
        print(f"[get_module_by_name] Interpreted '{layer_name}' as index -> '{name}'")
        return module

    # 3) try "module.<name>" — common after Opacus/DPDDP wrapping
    prefixed = f"_module.{layer_name}"
    if prefixed in modules_dict:
        return modules_dict[prefixed]

    # 4) if we already passed "module.xxx", also try stripping it
    if layer_name.startswith("_module."):
        stripped = layer_name[len("_module."):]
        if stripped in modules_dict:
            return modules_dict[stripped]

    raise KeyError(f"{layer_name=} not found. Keys: {list(modules_dict.keys())}...")


# hook to keep track of activations
def make_activation_hook(layer_name, args):
    def hook(module, input, output):
        #  Only log during training, not eval (test)
        if not module.training:
            return

        # Only log at selected epochs
        step = global_step["value"]
        if step not in args.TRACK_EPOCHS:
            return

        # Aggregate activations
        # assume [B, C, H, W] -> mean over batch + spatial dims -> [C]
        act = output.detach().mean(dim=(0, 2, 3)).cpu()
        layer_dict = activations.setdefault(layer_name, {})
        layer_dict.setdefault(step, []).append(act)

    return hook

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def argument_parser():
    parser = argparse.ArgumentParser()

    # Add all arguments upfront
    parser.add_argument('--dataset', default='Duke', choices=["Duke", "UMN"])

    parser.add_argument('--batch_size', default=8 , type=int)
    parser.add_argument('--num_iterations', default=100,type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--n_classes', default=9, type=int)
    parser.add_argument('--ffc_lambda', default=0, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--image_size', default='224', type=int)
    parser.add_argument('--model_name', default="unet",
                        choices=["unet", "y_net_gen", "y_net_gen_ffc", 'ReLayNet', 'UNetOrg', 'LFUNet', 'FCN8s',
                                 'NestedUNet','SimplifiedFCN8s','ConvNet'])
    parser.add_argument('--g_ratio', default=0.5, type=float)
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--seed', default=7, type=int)

    parser.add_argument('--in_channels', default=1, type=int)
    # Initially, do not set a default for image_dir
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--DPSGD', type=str2bool, default=False)
    parser.add_argument('--test', type=str2bool, default=True)
    # if save
    parser.add_argument('--model_should_be_saved', type=str2bool, default=False)
    parser.add_argument('--model_should_be_load', type=str2bool, default=False)
    parser.add_argument('--save_dir', default='./saved_models/', type=str, help="Directory to save the models")
    parser.add_argument('--epsilon', default=200, type=float, help="Privacy epsilon value for DPSGD")
    parser.add_argument('--morphology', default=True, type=str2bool, help="morphology")
    parser.add_argument('--operation',default='close', type=str, help="close, open or both")
    parser.add_argument('--kernel_size', default=5, type=int, help="kernel size")
    parser.add_argument('--learnable_radius',default=False,type=str2bool)
    parser.add_argument('--collect_activation',default=False,type=str2bool)
    parser.add_argument('--conditional_morph',default=False,type=str2bool)
    parser.add_argument('--use_morph',default=False,type=str2bool)
    parser.add_argument('--conditional_point',default="None",type=str,help="training,testing,validation,optional,None")
    parser.add_argument('--retinal_layer_wise', default=False, type=str2bool)
    parser.add_argument('--quantile_20_85', default=True, type=str2bool)
    parser.add_argument('--quantile_25_80', default=False, type=str2bool)
    parser.add_argument('--excluding_3_4_layer', default=True, type=str2bool)
    parser.add_argument('--policy_type', default='None')
    parser.add_argument('--max_grad_norm',default=1,type=float)
    parser.add_argument('--deep_supervision',default=False,type=str2bool)
    parser.add_argument(
            "--clipping",
            default="flat",
            type=str,
            choices=["None","flat", "automatic", "psac", "normalized_sgd"],
            help="Gradient clipping strategy for DP-SGD (flat, automatic, psac, normalized_sgd)",
        )
    #parser.add_argument('--excluding_background', default=True, type=str2bool, help="this is only useful when retinal layer wise is not active and we want to exclude/include background for morphology")
    parser.add_argument(
        "--TARGET_LAYERS",
        type=str,
        nargs="+",
        default=["all_conv"],  # e.g. ["0", "3", "6", "9"], or the real name of the layers
        help="List of layer names (from model.named_modules()) to track activations for",
    )
    parser.add_argument(
        "--TRACK_EPOCHS",
        type=int, nargs="+", default=[10, 50, 100, 200], help="Epochs at which to record activations", )

    parser.add_argument('--debug_morph_visual', type=str2bool, default=True,
                        help='Save morphology before/after visualizations during training')

    parser.add_argument('--debug_morph_every', type=int, default=10,
                        help='Save one visualization every N morphology events')

    parser.add_argument('--debug_morph_num_images', type=int, default=2,
                        help='Number of images to visualize from the batch')

    parser.add_argument('--debug_morph_dir', type=str, default='./morph_debug_events',
                        help='Directory to save morphology visualizations')
    # Parse the arguments
    args = parser.parse_args()
    # Dynamically set the image_dir based on the dataset argument
    if not args.image_dir:  # Only set image_dir if it wasn't explicitly provided
        if args.dataset == "UMN":
            args.image_dir = "/scicore/home/wagner0024/parsar0000/miniconda3/OCT_Sci/UMNData/"
        else:
            args.image_dir =  "/scicore/home/wagner0024/parsar0000/miniconda3/OCT_Sci/DukeData/"

    return args



def print_summary_table(header, row):
    """
    Print a one-row ASCII table with given header and row values.
    """
    # Convert everything to string once
    row_str = [str(x) for x in row]

    # Column widths = max(len(header), len(value))
    col_widths = [
        max(len(str(h)), len(v))
        for h, v in zip(header, row_str)
    ]

    # Build lines
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths))
    sep_line    = "-+-".join("-" * w for w in col_widths)
    value_line  = " | ".join(f"{v:<{w}}" for v, w in zip(row_str, col_widths))

    print("\n" + "=" * len(header_line))
    print("FINAL RUN SUMMARY")
    print(header_line)
    print(sep_line)
    print(value_line)
    print("=" * len(header_line) + "\n")



def compute_seg_loss(pred, label, criterion_seg, device):
    """
    pred: model output (either tensor or list/tuple of tensors)
    label: ground-truth [B,1,H,W]
    returns: (loss, last_pred_with_logits)
    """
    if isinstance(pred, (list, tuple)):
        # deep supervision
        weights = [0.1, 0.2, 0.3, 0.4]
        loss = 0
        for out, w in zip(pred, weights):
            loss = loss + w * criterion_seg(out, label.squeeze(1), device=device)
        last = pred[-1]
    else:
        loss = criterion_seg(pred, label.squeeze(1), device=device)
        last = pred

    return loss, last

# Differential Privacy-Based Random Morphology Selector
def dp_select_from_choices(choices, target_choice, privacy_budget):
    """
    Selects from a set of choices with probabilities inspired by differential privacy principles.

    Args:
        choices (list): A list of possible choices.
        target_choice (str): The specific choice to prioritize.
        privacy_budget (float): The differential privacy budget (\epsilon), controlling the randomness.

    Returns:
        str: Selected choice.
    """
    if target_choice not in choices:
        raise ValueError("Target choice must be in the list of choices.")

    # Calculate probabilities based on the formulas
    exp_eps = np.exp(privacy_budget)
    P_D_plus = (1 / (exp_eps + 1)) + ((exp_eps - 1) / (exp_eps + 1))
    P_D_minus = 1 / (exp_eps + 1)

    # Assign probabilities to choices
    probabilities = []
    for choice in choices:
        if choice == target_choice:
            probabilities.append(P_D_plus)
        else:
            probabilities.append(P_D_minus)

    # Normalize probabilities (to handle potential floating-point imprecision)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    # Randomly select a choice based on probabilities
    selected_choice = np.random.choice(choices, p=probabilities)
    print("selected choice:", selected_choice)
    return selected_choice


def colored_text(st):
    return '\033[91m' + st + '\033[0m'


def plot_examples(data_loader, model, device, num_examples=3):
    model.eval()  # Set the model to evaluation mode
    fig, axs = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))

    batch_processed = 0
    for imgs, masks in data_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():  # We do not need to compute gradients here
            preds = model(imgs)

        # Assuming the output is a softmax layer
        _, predicted_masks = torch.max(preds, dim=1)
        # Determine the maximum label for setting up the colormap
        max_label= max (np.max(masks.cpu().numpy()), np.max(predicted_masks.cpu().numpy()))
        cmap = plt.cm.get_cmap('viridis', max_label+1)

        for idx in range(imgs.size(0)):
            if batch_processed>= num_examples:
                break

            img=imgs[idx].cpu().numpy().squeeze()
            mask=masks[idx].cpu().numpy().squeeze()
            predicted_mask=predicted_masks[idx].cpu().numpy()

            axs[batch_processed, 0].imshow(img, cmap='gray')  # Assuming image is in the first channel
            axs[batch_processed, 0].set_title('Input Image')
            axs[batch_processed, 0].axis('off')

            axs[batch_processed, 1].imshow(mask, cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=max_label))
            axs[batch_processed, 1].set_title('Ground Truth Mask')
            axs[batch_processed, 1].axis('off')

            axs[batch_processed, 2].imshow(predicted_mask, cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=max_label))
            axs[batch_processed, 2].set_title('Predicted Mask')
            axs[batch_processed, 2].axis('off')

            batch_processed+=1
        if batch_processed >= num_examples:
            break

    plt.tight_layout()
    plt.show()


def segmentation_plots_test_morphology(val_loader, model, device,model_name,DPSGD, dataset, num_examples=5): # in this code, we already did not apply morphology

    folder_name = f"{'' if not DPSGD else 'DPSGD'}_{dataset}_images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    batch_processed=0
    model.eval()
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():  # We do not need to compute gradients here
            preds = model(imgs)

            """if args.morphology:
                refined_mask = apply_kornia_morphology_multiclass(
                    preds,
                    operation='erosion',
                    kernel_size=3,
                )
                #print("Loss value in original case", MAE(masks, preds, args.n_classes))
                #print("Loss value for refined case", MAE(masks, refined_mask, args.n_classes))
                _, refined_predicted_masks = torch.max(refined_mask, dim=1)"""

            _, predicted_masks = torch.max(preds, dim=1)

        indices = list(range(imgs.size(0)))
        random.shuffle(indices)


        for idx in indices:  # Use shuffled indices to select random images
            if batch_processed>= num_examples:
                break

            img=imgs[idx].cpu().numpy().squeeze()
            msk=masks[idx].cpu().numpy().squeeze()
            predicted_mask=predicted_masks[idx].cpu().numpy()
            """if args.morphology:
             refined_predicted_mask =refined_predicted_masks[idx].cpu().numpy()"""

            # Define colors for each layer (RGB tuples)
            layer_colors = {
                0: (0, 0, 0),  # Black for the modified 0
                1: (1, 0, 0),  # Red
                2: (0, 1, 0),  # Green
                3: (0, 0, 1),  # Blue
                4: (1, 1, 0),  # Yellow
                5: (1, 0, 1),  # Magenta
                6: (0, 1, 1),  # Cyan
                7: (1, 0.5, 0),  # Orange
                8: (0.8, 0.7, 0.6)  # Light Brown
            }

            def create_overlay(img, mask, layer_colors):
                overlay = np.zeros((img.shape[0], mask.shape[1], 3))
                for value, color in layer_colors.items():
                    overlay[mask == value] = color
                return overlay


            true_overlay = create_overlay(img, msk, layer_colors)
            predicted_overlay = create_overlay(img, predicted_mask, layer_colors)
            """if args.morphology:
                refined_predicted_overlay = create_overlay(img, refined_predicted_mask, layer_colors)"""

            original_rgb = np.stack([img] * 3, axis=-1)
            min_val = original_rgb.min()
            max_val = original_rgb.max()
            # Normalize based on observed min and max
            original_rgb = (original_rgb - min_val) / (max_val - min_val)

            true_combined = np.clip(0.7 * original_rgb + 0.3 * true_overlay, 0, 1)
            predicted_combined = np.clip(0.7 * original_rgb + 0.3 * predicted_overlay, 0, 1)
            """if args.morphology:
                refined_predicted_combined=np.clip(0.7 * original_rgb + 0.3 * refined_predicted_overlay, 0, 1)"""
            if DPSGD==True:
                state='DPSGD'
            else:
                state=''


            # Plotting
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))
            """if args.morphology:
             fig, axes = plt.subplots(1, 4, figsize=(30, 10))
            else:
                fig, axes = plt.subplots(1, 3, figsize=(30, 10))"""
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(true_combined)
            axes[1].set_title('True Mask')
            axes[1].axis('off')

            axes[2].imshow(predicted_combined)
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            """if args.morphology:
                axes[3].imshow(refined_predicted_combined)
                axes[3].set_title('Refined Predicted Mask')
                axes[3].axis('off')"""


            plt.axis('off')

            #file_path = os.path.join(folder_name, f'{model_name}_{state}_{dataset}_{idx}.pdf')
            #plt.savefig(file_path, format='pdf', bbox_inches='tight')
            plt.show()
            batch_processed += 1


def segmentation_plots(val_loader, model, device, model_name, DPSGD, dataset, num_examples=5):  # this version can work after applying morphological approach or normal cases without applying any morphological technique  (because after applying morphology then prediction of the model is the result of morphology)
    folder_name = f"{'' if not DPSGD else 'DPSGD'}_{dataset}_images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    batch_processed = 0
    model.eval()
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():  # We do not need to compute gradients here
            preds = model(imgs)
            _, predicted_masks = torch.max(preds, dim=1)

        indices = list(range(imgs.size(0)))
        random.shuffle(indices)

        for idx in indices:  # Use shuffled indices to select random images
            if batch_processed >= num_examples:
                break

            img = imgs[idx].cpu().numpy().squeeze()
            msk = masks[idx].cpu().numpy().squeeze()
            predicted_mask = predicted_masks[idx].cpu().numpy()

            # Define colors for each layer (RGB tuples)
            layer_colors = {
                0: (0, 0, 0),  # Black
                1: (1, 0, 0),  # Red
                2: (0, 1, 0),  # Green
                3: (0, 0, 1),  # Blue
                4: (1, 1, 0),  # Yellow
                5: (1, 0, 1),  # Magenta
                6: (0, 1, 1),  # Cyan
                7: (1, 0.5, 0),  # Orange
                8: (0.8, 0.7, 0.6)  # Light Brown
            }

            def create_overlay(img, mask, layer_colors):
                overlay = np.zeros((img.shape[0], mask.shape[1], 3))
                for value, color in layer_colors.items():
                    overlay[mask == value] = color
                return overlay

            true_overlay = create_overlay(img, msk, layer_colors)
            predicted_overlay = create_overlay(img, predicted_mask, layer_colors)

            original_rgb = np.stack([img] * 3, axis=-1)
            min_val = original_rgb.min()
            max_val = original_rgb.max()
            # Normalize based on observed min and max
            original_rgb = (original_rgb - min_val) / (max_val - min_val)

            true_combined = np.clip(0.7 * original_rgb + 0.3 * true_overlay, 0, 1)
            predicted_combined = np.clip(0.7 * original_rgb + 0.3 * predicted_overlay, 0, 1)

            if DPSGD == True:
                state = 'DPSGD'
            else:
                state = ''
            # Plotting

            fig, axes = plt.subplots(1, 3, figsize=(30, 10))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(true_combined)
            axes[1].set_title('True Mask')
            axes[1].axis('off')

            axes[2].imshow(predicted_combined)
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            plt.axis('off')
            # file_path = os.path.join(folder_name, f'{model_name}_{state}_{dataset}_{idx}.pdf')
            # plt.savefig(file_path, format='pdf', bbox_inches='tight')
            plt.show()
            batch_processed += 1


def visualize_batch(images, masks):
    batch_size = len(images)
    fig, ax = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))

    for i in range(batch_size):
        img = images[i].permute(1, 2, 0)  # Convert from CxHxW to HxWxC
        msk = masks[i].squeeze()  # Remove channel dim if it's there

        ax[i, 0].imshow(img)
        ax[i, 0].set_title('Input Image')
        ax[i, 0].axis('off')

        ax[i, 1].imshow(img)
        ax[i, 1].imshow(msk, alpha=0.3, cmap='jet')  # Overlay mask with transparency
        ax[i, 1].set_title('Overlay Mask')
        ax[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def append_morph_debug_csv(
    csv_path,
    *,
    epoch,
    conditional_point,
    retinal_layer_wise,
    stage,
    changed_ratio,
    fixed_ratio,
    worsened_ratio,
    dice_before,
    dice_after,
    seed,
    policy_type,
    excluding_3_4_layer,
    quantile_25_80,
    quantile_20_85
):
    """
    dice_before, dice_after: 1D numpy arrays of shape [C]
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    dice_before = np.asarray(dice_before, dtype=float)
    dice_after = np.asarray(dice_after, dtype=float)
    dice_gain = dice_after - dice_before

    row = {
        "epoch": epoch,
        "conditional_point": conditional_point,
        "retinal_layer_wise": retinal_layer_wise,
        "excluding_3_4_layer":excluding_3_4_layer,
        "quantile_25_80":quantile_25_80,
        "quantile_20_85":quantile_20_85,
        "stage": stage,
        "changed_ratio": float(changed_ratio),
        "fixed_ratio": float(fixed_ratio),
        "worsened_ratio": float(worsened_ratio),
        "dice_before_mean_no_bg": float(dice_before[1:].mean()),
        "dice_after_mean_no_bg": float(dice_after[1:].mean()),
        "dice_gain_mean_no_bg": float(dice_gain[1:].mean()),
        "seed":seed,
    }

    for c in range(len(dice_before)):
        row[f"dice_before_c{c}"] = float(dice_before[c])
        row[f"dice_after_c{c}"] = float(dice_after[c])
        row[f"dice_gain_c{c}"] = float(dice_gain[c])

    df_row = pd.DataFrame([row])

    if os.path.exists(csv_path):
        try:
            old = pd.read_csv(csv_path)
            new = pd.concat([old, df_row], ignore_index=True)
            #new.to_csv(csv_path, index=False)
        except:
            new=df_row
    else:
        new=df_row

    new.to_csv(csv_path, index=False)

def append_policy_debug_csv(
    csv_path,
    *,
    epoch,
    batch_idx,
    model_name,
    dataset,
    conditional_point,
    retinal_layer_wise,
    use_morph,
    policy_debug,
    seed,
    policy_type,
    excluding_3_4_layer,
    quantile_25_80,
    quantile_20_85,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row={
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "model_name": model_name,
        "dataset": dataset,
        "conditional_point": conditional_point,
        "retinal_layer_wise": retinal_layer_wise,
        "use_morph": bool(use_morph),
        "T_thin": policy_debug["T_thin"],
        "T_thick": policy_debug["T_thick"],
        "seed": seed,
        "policy_type":policy_type,
        "excluding_3_4_layer":excluding_3_4_layer,
        "quantile_25_80":quantile_25_80,
        "quantile_20_85":quantile_20_85

        }
    thickness_mean_c = policy_debug["thickness_mean_c"].numpy()
    for c,val in enumerate(thickness_mean_c):
        row[f"thickness_mean_c{c}"] = float(val)
    ops_by_class = policy_debug["ops_by_class"]
    for c, op in ops_by_class.items():
        row[f"op_c{c}"] = op
    df_row=pd.DataFrame([row])
    if os.path.exists(csv_path):
        try:
            old= pd.read_csv(csv_path)
            new=pd.concat([old,df_row],ignore_index=True)
        except:
            new= df_row
    else:
        new=df_row
    new.to_csv(csv_path,index=False)

def _gray_to_rgb(img_np):
    rgb = np.stack([img_np] * 3, axis=-1).astype(np.float32)
    mn, mx = rgb.min(), rgb.max()
    if mx > mn:
        rgb = (rgb - mn) / (mx - mn)
    else:
        rgb[:] = 0
    return rgb


def _make_color_mask(mask_np):
    layer_colors = {
        0: (0, 0, 0),        # black
        1: (1, 0, 0),        # red
        2: (0, 1, 0),        # green
        3: (0, 0, 1),        # blue
        4: (1, 1, 0),        # yellow
        5: (1, 0, 1),        # magenta
        6: (0, 1, 1),        # cyan
        7: (1, 0.5, 0),      # orange
        8: (0.8, 0.7, 0.6),  # light brown
    }
    h, w = mask_np.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    for c, color in layer_colors.items():
        overlay[mask_np == c] = color
    return overlay

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
def compute_true_and_clipped_grad_cosine(model, clipping_type="flat", max_grad_norm=1.0, eps=1e-12):
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

    if clipping_type == "flat":
        weights = (max_grad_norm / (norms + eps)).clamp(max=1.0)

    elif clipping_type=='automatic':
        r = max_grad_norm
        weights = max_grad_norm / (norms + 0.01)
    elif clipping_type=='normalized_sgd':
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


@torch.no_grad()
def save_morph_debug_visualization(
     imgs,
     masks,
     before_logits,
     after_logits,
     save_path,
     num_images=2,
     n_classes=9,
     title_prefix=""
 ):
     """
     imgs:         [B,1,H,W]
     masks:        [B,1,H,W] or [B,H,W]
     before_logits [B,C,H,W]
     after_logits  [B,C,H,W]

     Panels:
       1) Input
       2) Ground Truth
       3) Before Morph
       4) After Morph
       5) Changed Pixels
       6) Error Before
       7) Error After
       8) Improvement Map
          - Green  : fixed by morphology
          - Red    : worsened by morphology
          - Yellow : changed but still wrong
     """
     pred_before = before_logits.argmax(dim=1)
     pred_after = after_logits.argmax(dim=1)

     if masks.ndim == 4:
         gt = masks[:, 0]
     else:
         gt = masks

     changed = (pred_before != pred_after)
     err_before = (pred_before != gt)
     err_after = (pred_after != gt)

     fixed = (pred_before != gt) & (pred_after == gt)
     worsened = (pred_before == gt) & (pred_after != gt)
     changed_wrong = (pred_before != gt) & (pred_after != gt) & (pred_before != pred_after)

     b = min(num_images, imgs.size(0))
     fig, axes = plt.subplots(b, 8, figsize=(28, 3.8 * b))
     if b == 1:
         axes = np.expand_dims(axes, axis=0)

     for i in range(b):
         img_np = imgs[i, 0].detach().cpu().numpy()
         gt_np = gt[i].detach().cpu().numpy()
         pb_np = pred_before[i].detach().cpu().numpy()
         pa_np = pred_after[i].detach().cpu().numpy()

         changed_np = changed[i].detach().cpu().numpy().astype(np.float32)
         err_before_np = err_before[i].detach().cpu().numpy().astype(np.float32)
         err_after_np = err_after[i].detach().cpu().numpy().astype(np.float32)

         fixed_np = fixed[i].detach().cpu().numpy()
         worsened_np = worsened[i].detach().cpu().numpy()
         changed_wrong_np = changed_wrong[i].detach().cpu().numpy()

         img_rgb = _gray_to_rgb(img_np)

         gt_overlay = np.clip(0.7 * img_rgb + 0.3 * _make_color_mask(gt_np), 0, 1)
         pb_overlay = np.clip(0.7 * img_rgb + 0.3 * _make_color_mask(pb_np), 0, 1)
         pa_overlay = np.clip(0.7 * img_rgb + 0.3 * _make_color_mask(pa_np), 0, 1)

         # improvement map
         improve_map = np.zeros((*gt_np.shape, 3), dtype=np.float32)
         improve_map[fixed_np] = [0.0, 1.0, 0.0]          # green
         improve_map[worsened_np] = [1.0, 0.0, 0.0]      # red
         improve_map[changed_wrong_np] = [1.0, 1.0, 0.0] # yellow

         improve_overlay = np.clip(0.65 * img_rgb + 0.35 * improve_map, 0, 1)

         axes[i, 0].imshow(img_np, cmap="gray")
         axes[i, 0].set_title("Input", fontsize=10)
         axes[i, 0].axis("off")

         axes[i, 1].imshow(gt_overlay)
         axes[i, 1].set_title("GT", fontsize=10)
         axes[i, 1].axis("off")

         axes[i, 2].imshow(pb_overlay)
         axes[i, 2].set_title("Before", fontsize=10)
         axes[i, 2].axis("off")

         axes[i, 3].imshow(pa_overlay)
         axes[i, 3].set_title("After", fontsize=10)
         axes[i, 3].axis("off")

         axes[i, 4].imshow(changed_np, cmap="hot")
         axes[i, 4].set_title("Changed", fontsize=10)
         axes[i, 4].axis("off")

         axes[i, 5].imshow(err_before_np, cmap="hot")
         axes[i, 5].set_title("Err Before", fontsize=10)
         axes[i, 5].axis("off")

         axes[i, 6].imshow(err_after_np, cmap="hot")
         axes[i, 6].set_title("Err After", fontsize=10)
         axes[i, 6].axis("off")

         axes[i, 7].imshow(improve_overlay)
         axes[i, 7].set_title("Improve Map", fontsize=10)
         axes[i, 7].axis("off")

     changed_ratio = changed.float().mean().item()
     fixed_ratio = fixed.float().mean().item()
     worsened_ratio = worsened.float().mean().item()
     pb_np = pred_before.cpu().numpy()
     pa_np = pred_after.cpu().numpy()
     gt_np = gt.cpu().numpy()

     dice_before = np.array(compute_dice(gt_np, pb_np,num_classes=n_classes))
     dice_after  = np.array(compute_dice(gt_np, pa_np,num_classes=n_classes))

     dice_before_mean = dice_before[1:].mean()
     dice_after_mean  = dice_after[1:].mean()
     dice_gain = dice_after_mean - dice_before_mean
     dice_delta  = dice_after - dice_before
     print("[MORPH DEBUG] dice per class delta :", np.round(dice_delta, 4))

     print(
         f"[MORPH DEBUG] changed={changed_ratio:.5f} "
         f"fixed={fixed_ratio:.5f} "
         f"worsened={worsened_ratio:.5f} "
         f"dice_before={dice_before_mean:.4f} "
         f"dice_after={dice_after_mean:.4f} "
         f"dice gain={dice_gain:.4f}"
     )

     """plt.suptitle(
         f"{title_prefix} | changed={changed_ratio:.5f} "
         f"| fixed={fixed_ratio:.5f} "
         f"| worsened={worsened_ratio:.5f} "
         f"| dice_before={dice_before_mean:.4f} "
         f"| dice_after={dice_after_mean:.4f} "
         f"| gain={dice_gain:.4f}",
         fontsize=12
     )"""
     plt.tight_layout()
     os.makedirs(os.path.dirname(save_path), exist_ok=True)
     plt.savefig(save_path, dpi=220, bbox_inches="tight")
     plt.close()
     return {
            "changed_ratio": changed_ratio,
            "fixed_ratio": fixed_ratio,
            "worsened_ratio": worsened_ratio,
            "dice_before": dice_before,
            "dice_after": dice_after,}

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_files(path, ext):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(ext)]


def save_results_to_csv(args, model,file_name, max_grad_norm, noise_multiplier,
                        model_name, learning_rate, batch_size,
                        training_losses, validation_losses,
                        dice_all, validation_dice_scores,
                        privacy_epsilons, iterations, dataset,
                        mae, per_layer_all_list_MAE,hd95_all,wall_time_total_s=None,
                                                    wall_time_train_s=None,
                                                    wall_time_val_s=None,
                                                    wall_time_test_s=None,
                                                    gpu_name=None,seed=None,policy_type="None",quantile_25_80=False,quantile_20_85=True):

    # Create output directory
    out_dir = "csv_results"
    os.makedirs(out_dir, exist_ok=True)
    model_name_tagged = model_name
    if args.morphology:
        if args.learnable_radius:
            model_name_tagged += "_LearnR"
        else:
            model_name_tagged += f"_FixedR{args.kernel_size}"
    else:
        model_name_tagged += "_NoMorph"
    if args.DPSGD:
        clipping_strat=args.clipping
    else:
        clipping_strat=None
    # --- unified header, always the same ---
    header = [
        "Model_Name", "Learning_Rate", "Batch_Size",
        "Training_Loss", "Validation_Loss",
        "dice_all", "Validation_Dice",
        "Privacy_Epsilons", "max_grad_norm",
        "noise_multiplier", "iterations",
        "dataset", "mae", "per_layer_all_list",
        "Operation", "Kernel",
        "conditional_morph", "conditional_point", "initial_use_morph","final_use_morph","retinal_layer_wise","hd95_all","best_hd95","last_hd95",
        "wall_time_total_s", "wall_time_train_s", "wall_time_val_s", "wall_time_test_s",
        "gpu_name",
        "clipping_strategy",
        "seed",
        "policy_type",
        "excluding_3_4_layer",
        "Qu_25_80",
        "Qu_20_85",

    ]
    valid_hd95 = [x for x in hd95_all if not np.isnan(x)]
    # fill Operation/Kernel even for non-morph models
    op = args.operation if args.morphology else "None"
    ks = args.kernel_size if args.morphology else "None"
    cond_point = args.conditional_point if args.conditional_morph else "None"
    # last element of each list
    row_data = [
        model_name_tagged,
        learning_rate,
        batch_size,
        training_losses[-1] if training_losses else None,
        validation_losses[-1] if validation_losses else None,
        dice_all[-1] if dice_all else None,
        validation_dice_scores[-1] if validation_dice_scores else None,
        privacy_epsilons if privacy_epsilons else None,
        max_grad_norm,
        noise_multiplier,
        iterations,
        dataset,
        mae[-1] if isinstance(mae, (list, tuple)) else mae,
        per_layer_all_list_MAE[-1] if per_layer_all_list_MAE else None, # last list of results
        op,
        ks,
        args.conditional_morph,
        cond_point,
        args.use_morph,
        bool(getattr(model, "use_morph", getattr(args, "use_morph", False))),
        args.retinal_layer_wise,
        str(hd95_all),
        min(valid_hd95) if len(valid_hd95) > 0 else np.nan,
        hd95_all[-1] if len(hd95_all) > 0 else np.nan,
        wall_time_total_s,
        wall_time_train_s,
        wall_time_val_s,
        wall_time_test_s,
        gpu_name,
        clipping_strat,
        seed,
        policy_type,
        args.excluding_3_4_layer,
        quantile_25_80,
        quantile_20_85,

    ]
    path = os.path.join(out_dir, file_name)
    file_exists = os.path.isfile(path)
    try:
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row_data)

        file_path = os.path.abspath(path)
        print(f"Results saved to {file_path}")
        print_summary_table(header, row_data)
    except Exception as e:
        print(f"Failed to save results to CSV: {e}")


def eval(args,model_name,val_loader, criterion, model, n_classes,
         dice_s=True, device="cuda", im_save=False,
         use_morph: bool | None = None, epoch=None, mode="val"):

    model.eval()
    counter = 0
    dice = 0
    mae = 0
    total_loss = 0
    total_hd95 = 0.0
    hd95_count = 0
    dice_all = np.zeros(n_classes)
    per_layer_all_MAE = np.zeros(n_classes)
    if n_classes == 9:
        valid_classes = list(range(1, n_classes - 1))  # retinal layers only
    elif n_classes == 2:
        valid_classes = [1]  # fluid only

    if float(args.epsilon).is_integer():
            eps_str = f"eps{int(args.epsilon)}"  # 2.0 -> "eps2"
    else:
            eps_str = f"eps{str(args.epsilon)}"  # 2.5 -> "eps2.5" (or replace '.' if you want)

    epsilon_val = eps_str if args.DPSGD else ""

    with torch.no_grad():
        for img, label in tqdm.tqdm(val_loader):
            img = img.to(device)
            label = label.to(device)
            label = label.squeeze(1)
            label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes)
            # key change: pass use_morph to the model
            if use_morph is None:
                pred = model(img)
            else:
                pred = model(img, use_morph=use_morph)
            #----------------------------------------
            # collect tickness information when retinal_layer_wise is true
            target_model = model._module if hasattr(model, "_module") else model
            if (
                use_morph
                and args.morphology
                and args.retinal_layer_wise
                and getattr(target_model, "last_policy_debug", None) is not None):

                policy_csv_path = os.path.join(
                    args.debug_morph_dir,
                    args.dataset,
                    model_name,
                    f"cond_{args.conditional_point}",
                    f"rlw_{args.retinal_layer_wise}",
                    f"excl_three_four_{args.excluding_3_4_layer}",
                    f"{mode}_policy_debug_seed_clipping{args.clipping}_seed{args.seed}_eps{epsilon_val}.csv"   # 👈 val / test
                )

                append_policy_debug_csv(
                    policy_csv_path,
                    epoch=epoch,
                    batch_idx=counter,
                    model_name=model_name,
                    dataset=args.dataset,
                    conditional_point=args.conditional_point,
                    retinal_layer_wise=args.retinal_layer_wise,
                    use_morph=use_morph,
                    policy_debug=target_model.last_policy_debug,
                    seed=args.seed,
                    policy_type=args.policy_type,
                    excluding_3_4_layer=args.excluding_3_4_layer,
                    quantile_25_80= args.quantile_25_80,
                    quantile_20_85= args.quantile_20_85,

                )


            #-----------------------------------------------
            loss, pred = compute_seg_loss(pred, label, criterion, device)
            total_loss += loss.item()

            max_val, idx = torch.max(pred, 1)
            pred_seg = idx.detach().cpu().numpy()
            label_seg = label.detach().cpu().numpy()
            ret = compute_dice(label_seg, pred_seg,num_classes=n_classes)
            pa = compute_pa(label_seg, pred_seg,num_classes=n_classes)
            # ---- HD95 per sample ----
            batch_hd95 = []
            for b in range(pred_seg.shape[0]): # just for retinal layers
                hd95_val = hd95_multiclass(
                    pred_seg[b],
                    label_seg[b],
                    classes=valid_classes)   # retinal layers only

                if not np.isnan(hd95_val):
                    batch_hd95.append(hd95_val)

            if len(batch_hd95) > 0:
                total_hd95 += float(np.mean(batch_hd95))
                hd95_count += 1

            pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
            if dice_s:
                d1, d2 = per_class_dice(pred_oh, label_oh, n_classes)
                dice += d1
                dice_all += d2

            label_mae = torch.nn.functional.one_hot(label, num_classes=n_classes).squeeze()
            label_mae = label_mae.permute(0, 3, 1, 2)

            init_mae, per_layer_MAE = MAE_New(label_mae, pred, n_classes=n_classes,classes=valid_classes)
            mae += init_mae.item()
            per_layer_all_MAE += per_layer_MAE
            counter += 1


        loss = total_loss / counter
        dice_all = dice_all / counter
        per_layer_all_MAE = per_layer_all_MAE / counter
        mae = mae / counter # this is already 7-layer MAE (thanks to classes=[1..7])
        mean_hd95 = total_hd95 / hd95_count if hd95_count > 0 else np.nan


        retinal_mean_dice = dice_all[valid_classes].mean()
        non_bg_mean_dice = dice_all[1:].mean() # 8-layer (if including fluid)
        mean_dice_all = dice_all.mean()

        print(
                "Validation loss:", loss,
                "\n  Mean Dice (all classes):", mean_dice_all,
                "\n  Mean Dice (layers 1–7 only):", retinal_mean_dice,
                "\n  Mean Dice (classes 1–8, no bg):", non_bg_mean_dice,
                "\n  Dice All:", dice_all, # all classes including background
                "\n  MAE (layers 1–7):", mae,
                "\n  per layer MAE (1–7):", per_layer_all_MAE[valid_classes],
                "\n  HD95 (layers 1–7):", mean_hd95
            )

        return retinal_mean_dice, loss, dice_all, mae, per_layer_all_MAE, mean_hd95



def choose_use_morph(args, model, t, img, label, criterion_seg, device):
    """
    Returns True/False depending on args policy.
    """
    # no morphology at all
    if not args.morphology:
        return False
    # unconditional morphology: always on
    if not args.conditional_morph:
        return True
    if (t % 5 == 0): # to control time complexity
        # conditional policies
        if args.conditional_point == "training":
            print("conditional point is training ")
            # example: compare losses
            with torch.no_grad():
                pred_nm = model(img, use_morph=False)
                loss_nm, _ = compute_seg_loss(pred_nm, label, criterion_seg, device)

                pred_m = model(img, use_morph=True)
                loss_m, _ = compute_seg_loss(pred_m, label, criterion_seg, device)

            return loss_m.item() < loss_nm.item()

        # conditional decided elsewhere (validation), so use model.use_morph
        if hasattr(model, "use_morph"):
            return bool(model.use_morph)

        # fallback
    return bool(args.use_morph)

def train(args):
    if not args.retinal_layer_wise: #  to prevent any conflict
        args.policy_type="None"

    import os
    t_total_start = time.time()
    t_train_acc = 0.0
    t_val_acc = 0.0
    t_test_acc = 0.0

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    device = args.device
    n_classes = args.n_classes
    model_name = args.model_name
    # to handle deep supervsion naming for NestedUNet
    if model_name=="NestedUNet" and args.deep_supervision:
        model_name="deepsuper_NestedUNet"
        print("model name updated based on the deep supervision status")

    learning_rate = args.learning_rate
    ratio = args.g_ratio
    data_path = args.image_dir
    iterations = args.num_iterations
    img_size = args.image_size
    batch_size = args.batch_size
    test=args.test
    morph_event_counter = 0 # will be used for debugging the morphology impact
    training_losses = []
    validation_losses = []
    dice_all_list=[]
    per_layer_all_list_MAE=[]
    mae_all=[]
    validation_dice_scores = []
    hd95_all = []
    gradient_cosines = []
    gradient_cosine_epochs = []
    gradient_cosine_batches = []
    max_consecutive_epochs_without_improvement = 10
    consecutive_epochs_without_improvement = 0
    best_test_loss = float("inf")

    # calculating len of data for delta
    train_data_path = os.path.join(data_path, 'train')
    train_data_path_images = os.path.join(train_data_path, 'images')
    images_files = os.listdir(train_data_path_images)
    image_files = [file for file in images_files if file.endswith(('.npy'))]
    number_of_images = len(image_files)

    criterion_seg = CombinedLoss()
    criterion_ffc = FocalFrequencyLoss()
    if args.morphology:
        final_save_dir=os.path.join(args.save_dir, 'morphology_models')
    else:
        final_save_dir=os.path.join(args.save_dir, 'models')

    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir)

    # --- tag for learnable vs fixed morphology ---
    if args.morphology:
        if args.learnable_radius:
            morph_tag = "LearnR"          # learnable radius
        else:
            morph_tag = f"FixedR{args.kernel_size}"  # fixed radius = kernel_size
    else:
        morph_tag = "NoMorph"

    if args.DPSGD:
        if args.morphology:
            save_name = os.path.join(
                final_save_dir,
                f"{model_name}_{args.dataset}_DPSGD_"
                f"{args.num_iterations}_{args.batch_size}_{args.epsilon}_"
                f"retinal_layer_wise_{args.retinal_layer_wise}_excl_three_four_{args.excluding_3_4_layer}_policy_type_{args.policy_type}_"
                f"{args.operation}_{args.kernel_size}_{morph_tag}_seed{args.seed}.pt"
            )
        else:
            save_name = os.path.join(
                final_save_dir,
                f"{model_name}_{args.dataset}_DPSGD_"
                f"{args.num_iterations}_{args.batch_size}_{args.epsilon}_{morph_tag}_seed{args.seed}.pt"
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

    max_dice = 0
    best_test_dice = 0
    best_iter = 0
    if args.morphology:
        model= get_model(args.model_name, ratio=ratio, num_classes=n_classes,morphology=args.morphology,operation=args.operation,kernel_size=args.kernel_size,learnable_radius=args.learnable_radius,use_morph=args.use_morph,retinal_layer_wise=args.retinal_layer_wise,deep_supervision=args.deep_supervision,policy_version=args.policy_type,excluding_3_4_layer=args.excluding_3_4_layer,quantile_20_85=args.quantile_20_85,quantile_25_80=args.quantile_25_80).to(device)
    else:
        model = get_model(args.model_name, ratio=ratio, num_classes=n_classes,use_morph=False,deep_supervision=args.deep_supervision).to(device)
    print(model_name)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate,
                                 weight_decay=args.weight_decay)

    train_loader, val_loader, test_loader, _, _, _ = get_data(data_path, img_size, batch_size)

    #for images, labels in train_loader:
        #visualize_batch(images, labels)
        #break

    if args.model_should_be_load:
        if os.path.exists(save_name):
            checkpoint = torch.load(save_name)
            state_dict = checkpoint['model_state_dict']

            # Strip the `_module.` prefix if it exists
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('_module.', '')  # Remove the _module prefix
                new_state_dict[new_key] = state_dict[key]

            model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iteration = checkpoint['iteration'] + 1
            training_losses.append(checkpoint['training_losses'])
            validation_losses.append(checkpoint['validation_losses'])
            validation_dice_scores.append(checkpoint['validation_dice_scores'])
            best_test_loss = checkpoint['best_test_loss']
            dice_all_list = checkpoint['dice_all_list']  # per-class Dice history
            per_layer_all_list_MAE = checkpoint['per_layer_all_list_MAE']  # per-class MAE history (as strings)
            mae_all = checkpoint['mae_all']
            consecutive_epochs_without_improvement = checkpoint['consecutive_epochs_without_improvement']
            print(f"Resuming training from iteration {start_iteration}")

    else:
        start_iteration = 1

    # delta = 1 / (number_of_images ** 1.1)
    if args.DPSGD==True:
        delta = 1e-5
        privacy_engine = PrivacyEngine()
        noise_multiplier = 1
        max_grad_norm = args.max_grad_norm
        model_name=f"{model_name}_DPSGD"

        if args.clipping == "flat":
            # Don't pass clipping argument for default behavior
            model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=args.epsilon,
                target_delta=delta,
                epochs=iterations,
                max_grad_norm=max_grad_norm,
            )
        else:
            # Pass clipping argument for advanced strategies
            model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=args.epsilon,
                target_delta=delta,
                epochs=iterations,
                max_grad_norm=max_grad_norm,
                clipping=args.clipping,
            )

        privacy_epsilons = []
    else:
        noise_multiplier = 'None'
        max_grad_norm = 'None'
        delta='None'
        model_name=model_name


    if float(args.epsilon).is_integer():
        eps_str = f"eps{int(args.epsilon)}"  # 2.0 -> "eps2"
    else:
        eps_str = f"eps{str(args.epsilon)}"  # 2.5 -> "eps2.5" (or replace '.' if you want)

    epsilon_val = eps_str if args.DPSGD else ""
    save_dir_activations = f"./activation_logs_oct/{args.dataset}/{model_name}_{epsilon_val}"
    os.makedirs(save_dir_activations, exist_ok=True)

    handles = []
    if args.collect_activation==True:
        if len(args.TARGET_LAYERS) == 1 and args.TARGET_LAYERS[0].lower() == "all_conv":
            # auto: all Conv2d layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    h = module.register_forward_hook(make_activation_hook(name, args))
                    handles.append(h)
                    print(f"Registered hook on conv layer: {name}")
        else:
            # manual list
            for layer_name in args.TARGET_LAYERS:
                layer_module = get_module_by_name(model, layer_name)
                h = layer_module.register_forward_hook(make_activation_hook(layer_name, args))
                handles.append(h)
                print(f"Registered hook on: {layer_name}")

        for name, p in model.named_parameters():  # WARNING: this is just for debugging
            if "logit_radius" in name:
                print("FOUND logit_radius in model:", name, "requires_grad =", p.requires_grad)
        radius_history=[]

    # training

    model.use_morph = args.use_morph  # default starting state
    for t in range(start_iteration, iterations+1):
        saved_morph_this_epoch = False
        global_step["value"] = t
        print("t iteration: ", t)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_train_epoch_start = time.time()
        model.train()
        total_loss = 0
        total_samples = 0

        if args.DPSGD:
                from opacus.utils.batch_memory_manager import BatchMemoryManager
                loader_cm = BatchMemoryManager(
                    data_loader=train_loader,
                    max_physical_batch_size=args.batch_size,
                    optimizer=optimizer,
                )
        else:
                from contextlib import nullcontext
                loader_cm = nullcontext(train_loader)

        with loader_cm as epoch_train_loader:
            for batch_idx, (img, label) in enumerate(epoch_train_loader):
                img = img.to(device)
                label = label.to(device)
                label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes).squeeze()
                # making decision about morphology
                use_morph = choose_use_morph(args, model, t, img, label, criterion_seg, device)

                ###-------------------------------------------
                # decide whether THIS actual morphology event should be captured
                will_capture_this_batch = (
                     args.debug_morph_visual
                     and use_morph
                     and ((morph_event_counter + 1) % args.debug_morph_every == 0))

                target_model= model._module if hasattr(model, "_module") else model # to be able to handle dpsgd
                if hasattr(target_model, "store_morph_debug"):
                    target_model.store_morph_debug = will_capture_this_batch
                pred_raw = model(img, use_morph=use_morph)
                # save operations during training for retinal layer wise case
                if (
                    args.morphology
                    and use_morph
                    and args.retinal_layer_wise
                    and getattr(target_model, "last_policy_debug", None) is not None
                    and ((morph_event_counter + 1) % args.debug_morph_every == 0)): # if you want to save the operations for all batches this can be commented

                    policy_csv_path = os.path.join(
                        args.debug_morph_dir,
                        args.dataset,
                        model_name,
                        f"cond_{args.conditional_point}",
                        f"rlw_{args.retinal_layer_wise}",
                        f"excl_three_four_{args.excluding_3_4_layer}",
                        f"train_policy_debug_seed_clipping_{args.clipping}_seed{args.seed}_eps{epsilon_val}.csv"
                    )

                    append_policy_debug_csv(
                        policy_csv_path,
                        epoch=t,
                        batch_idx=batch_idx,
                        model_name=model_name,
                        dataset=args.dataset,
                        conditional_point=args.conditional_point,
                        retinal_layer_wise=args.retinal_layer_wise,
                        use_morph=use_morph,
                        policy_debug=target_model.last_policy_debug,
                        seed=args.seed,
                        policy_type=args.policy_type,
                        excluding_3_4_layer=args.excluding_3_4_layer,
                        quantile_25_80=args.quantile_25_80,
                        quantile_20_85=args.quantile_20_85,

                    )

                # Save visualization only when morphology was REALLY applied in this forward
                if args.debug_morph_visual and use_morph:
                    morph_event_counter += 1
                    if will_capture_this_batch:
                            print(f"[DEBUG A] morphology event #{morph_event_counter} captured")
                    if will_capture_this_batch and getattr(target_model, "last_morph_debug", None) is not None:
                        debug_dir = os.path.join(
                            args.debug_morph_dir,
                            args.dataset,
                            model_name,
                            f"cond_{args.conditional_point}",
                            f"rlw_{args.retinal_layer_wise}"
                        )
                        os.makedirs(debug_dir, exist_ok=True)

                        stage = target_model.last_morph_debug["stage"]

                        save_path = os.path.join(
                            debug_dir,
                            f"ep{t:04d}_seed{args.seed}_policy{args.policy_type}_rlw{int(args.retinal_layer_wise)}_excl_three_four{args.excluding_3_4_layer}_cond{args.conditional_point}_stage-{stage}_ev{morph_event_counter:06d}.pdf"
                        )

                        stats=save_morph_debug_visualization(
                            imgs=img.detach().cpu(),
                            masks=label.detach().cpu(),
                            before_logits=target_model.last_morph_debug["before"],
                            after_logits=target_model.last_morph_debug["after"],
                            save_path=save_path,
                            num_images=args.debug_morph_num_images,
                            n_classes=n_classes,
                            title_prefix=(
                                f"epoch={t} | cond={args.conditional_point} | "
                                f"rlw={args.retinal_layer_wise} | "
                                f"stage={target_model.last_morph_debug['stage']}"
                            )
                        )
                        csv_path = os.path.join(
                            args.debug_morph_dir,
                            args.dataset,
                            model_name,
                            f"cond_{args.conditional_point}",
                            f"rlw_{args.retinal_layer_wise}",
                            f"excl_three_four_{args.excluding_3_4_layer}",
                            f"morph_debug_stats.csv"
                        )

                        append_morph_debug_csv(
                            csv_path,
                            epoch=t,
                            conditional_point=args.conditional_point,
                            retinal_layer_wise=args.retinal_layer_wise,
                            stage=target_model.last_morph_debug["stage"],
                            changed_ratio=stats["changed_ratio"],
                            fixed_ratio=stats["fixed_ratio"],
                            worsened_ratio=stats["worsened_ratio"],
                            dice_before=stats["dice_before"],
                            dice_after=stats["dice_after"],
                            seed= args.seed,
                            policy_type= args.policy_type,
                            excluding_3_4_layer=args.excluding_3_4_layer,
                            quantile_25_80=args.quantile_25_80,
                            quantile_20_85=args.quantile_20_85,

                        )

                        print(f"[DEBUG] Saved morphology visualization to: {save_path}")
                        print(f"[DEBUG] Appended morphology stats to: {csv_path}")


                    elif will_capture_this_batch:
                            print("[DEBUG B] wanted to capture, but last_morph_debug is None")

                #---------------------------------------------

                loss, pred = compute_seg_loss(pred_raw, label, criterion_seg, device)

                ##  for debuging (
                """with torch.no_grad():
                    target = label.squeeze(1).long().to(device)
                    ce_part = criterion_seg.cross_entropy_loss(pred, target)

                    pred_soft = F.softmax(pred, dim=1)
                    dice_part = criterion_seg.dice_loss(pred_soft, target)

                print(f"CE: {ce_part.item():.4f}, Dice: {dice_part.item():.4f}, Total: {loss.item():.4f}")"""
                #)
                max_val, idx = torch.max(pred, 1)
                pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
                pred_oh = pred_oh.permute(0, 3, 1, 2)
                #label_oh = label_oh.permute(0, 3, 1, 2)
                """loss = criterion_seg(pred, label.squeeze(1), device=device) + args.ffc_lambda * criterion_ffc(pred_oh,
                                                                                                           label_oh)"""

                optimizer.zero_grad() #zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
                loss.backward()

                # WARNING: this is just for debugging
                # Debug: check gradient of the learnable radius
                for name, param in model.named_parameters():
                    if "morph_layer.logit_radius" in name or name.endswith("logit_radius"):
                        print("FOUND", name, "requires_grad =", param.requires_grad)
                        print("  grad:", param.grad)

                #------------------------------------debugging for checking if morphology function has real impact
                """if (t%10==0) & args.morphology:
                    with torch.no_grad():
                       try:
                           out_nomorph = model(img, use_morph=False)
                           out_morph   = model(img, use_morph=True)

                           pred_nomorph = out_nomorph.argmax(dim=1)
                           pred_morph   = out_morph.argmax(dim=1)

                           changed_pixels = (pred_nomorph != pred_morph).float().mean().item()
                           logit_diff = (out_nomorph - out_morph).abs().mean().item()
                           print(f"[DEBUG] changed_pixels={changed_pixels:.6f}, mean_abs_logit_diff={logit_diff:.6f}")
                       except:
                           print("There was an issue with DEBUGGING")"""
                #-----------------------------------------------------
                #---------------------------------------------------------------
                # cosin similarity before and after applying clipping_strategy
                if args.DPSGD and (batch_idx % 10 == 0):
                    try:
                        stats = compute_true_and_clipped_grad_cosine(
                            model,
                            clipping_type=args.clipping,
                            max_grad_norm=args.max_grad_norm,
                        )
                        gradient_cosines.append(stats["cosine"])
                        gradient_cosine_epochs.append(t)
                        gradient_cosine_batches.append(batch_idx)

                        print(
                            f"[GRAD ALIGN] epoch={t} batch={batch_idx} "
                            f"clip={args.clipping} "
                            f"cos={stats['cosine']:.6f} "
                            f"true_norm={stats['true_grad_norm']:.6f} "
                            f"method_norm={stats['method_grad_norm']:.6f}"
                        )
                    except Exception as e:
                        print(f"[GRAD ALIGN ERROR] {e}")
                # ------------------------------------------------------------
                optimizer.step()
                total_loss += loss.item() * img.size(0)
                total_samples += img.size(0)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_train_acc += (time.time() - t_train_epoch_start)
        average_loss = total_loss / total_samples
        training_losses.append(average_loss)

        if args.DPSGD==True:
            epsilon = privacy_engine.get_epsilon(delta)
            privacy_epsilons.append(epsilon)

            print(
                f"\tTrain Epoch: [{t + 1}/{iterations}] \t"
                f"Train Loss: {np.mean(average_loss):.6f} "
                f"(ε = {epsilon:.2f}, δ = {delta})"
            )
        else:
            print(
                f"\tTrain Epoch: [{t + 1}/{iterations}] \t"
                f"Train Loss: {np.mean(average_loss):.6f} ")

        #if t % 20 == 0:  # Every 20 epochs
            #plot_examples(val_loader, model, device)
            #segmentation_plots(val_loader, model, device, model_name, DPSGD=args.DPSGD, dataset=args.dataset, num_examples=1)


        if (t % 5 == 0):  # e.g., every 5 epochs
            print("Validation")
            t_val_start = time.time()
            print(f"epoch number:{t}")
            print("Validation")
            print(f"epoch number:{t}")

            if args.morphology and  args.conditional_morph and args.conditional_point=="validation":
                print("applying validation conditional point")
                # 1) evaluate with morphology ON
                print("\033[94m[VAL] Testing with morphology ON\033[0m")
                dice_m, val_loss_m, dice_all_m, mae_m, per_layer_all_MAE_m,mean_hd95_m = eval(args,model_name,
                    val_loader,
                    criterion_seg,
                    model,
                    dice_s=True,
                    n_classes=n_classes,
                    device=device,
                    use_morph=True,
                    epoch=t,
                     mode="val"
                )

                # 2) evaluate with morphology OFF
                print("\033[94m[VAL] Testing with morphology OFF\033[0m")
                dice_nm, val_loss_nm, dice_all_nm, mae_nm, per_layer_all_MAE_nm,mean_hd95_nm = eval(args,model_name,
                    val_loader,
                    criterion_seg,
                    model,
                    dice_s=True,
                    n_classes=n_classes,
                    device=device,
                    use_morph=False,
                    epoch=t,
                    mode="val"
                )

                # 3) compare validation loss and choose
                if val_loss_m < val_loss_nm and  dice_m > dice_nm:
                    model.use_morph = True
                    print(
                        f"\033[91m[VAL] Using morphology for next epochs (loss_m={val_loss_m:.4f} < loss_nm={val_loss_nm:.4f})\033[0m")
                    # log "chosen" stats
                    dice = dice_m
                    validation_loss = val_loss_m
                    dice_all = dice_all_m
                    mae = mae_m
                    per_layer_all_MAE = per_layer_all_MAE_m
                    mean_hd95 = mean_hd95_m
                else:
                    model.use_morph = False
                    print(
                        f"\033[91m[VAL] Disabling morphology for next epochs (loss_nm={val_loss_nm:.4f} <= loss_m={val_loss_m:.4f})\033[0m")
                    dice = dice_nm
                    validation_loss = val_loss_nm
                    dice_all = dice_all_nm
                    mae = mae_nm
                    per_layer_all_MAE = per_layer_all_MAE_nm
                    mean_hd95 = mean_hd95_nm

            else:
                # original behavior: single eval with current state
                dice, validation_loss, dice_all, mae, per_layer_all_MAE,mean_hd95 = eval(args,model_name,
                    val_loader,
                    criterion_seg,
                    model,
                    dice_s=True,
                    n_classes=n_classes,
                    device=device,
                    epoch=t,
                    mode="val"
                )
            #segmentation_plots(val_loader, model, device, model_name, args.DPSGD, args.dataset)
            validation_losses.append(validation_loss)
            validation_dice_scores.append(dice.item())
            mae_all.append(mae)
            hd95_all.append(mean_hd95)
            dice_all_list.append(dice_all)
            mae_all_str = str(per_layer_all_MAE.tolist()) # to make it possible to work with ast
            per_layer_all_list_MAE.append(mae_all_str)
            t_val_acc += (time.time() - t_val_start)

            # print("Expert 1 - Test")
            # dice_test = eval(test_loader, criterion_seg, model, n_classes=n_classes)
            #------------------
            # saving
            if dice > max_dice:
                max_dice = dice
                best_iter = t
                # best_test_dice = dice_test
                # torch.save(model, save_name)
                if args.model_should_be_saved:
                    if args.DPSGD == True:
                        print(colored_text("Updating model, epoch: "), t)
                        torch.save({
                            'iteration': t,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_losses': training_losses,
                            'validation_losses': validation_losses,
                            'validation_dice_scores': validation_dice_scores,
                            'best_test_loss': best_test_loss,
                            'dice_all_list': dice_all_list,
                            'per_layer_all_list':per_layer_all_list_MAE,
                            'mae_all':mae_all,
                            'hd95_all': hd95_all,
                            'consecutive_epochs_without_improvement': consecutive_epochs_without_improvement,
                            'privacy_epsilons': privacy_epsilons,

                        }, save_name)
                    else:
                        print(colored_text("Updating model, epoch: "), t)
                        torch.save({
                            'iteration': t,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_losses': training_losses,
                            'validation_losses': validation_losses,
                            'validation_dice_scores': validation_dice_scores,
                            'best_test_loss': best_test_loss,
                            'dice_all_list': dice_all_list,
                            'per_layer_all_list':per_layer_all_list_MAE,
                            'mae_all':mae_all,
                            'hd95_all': hd95_all,
                            'consecutive_epochs_without_improvement': consecutive_epochs_without_improvement,
                        }, save_name)


            if validation_loss < best_test_loss : # if there is any improvement
                best_test_loss= validation_loss
                consecutive_epochs_without_improvement = 0  # reset
            else:
                consecutive_epochs_without_improvement +=1
            if consecutive_epochs_without_improvement >= max_consecutive_epochs_without_improvement:
                print(
                    f"Stopping training due to lack of improvement for {max_consecutive_epochs_without_improvement} epochs.")
                break  # Exit the training loop

            # lr scheduler
            """scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=15)
            # after validation
            scheduler.step(dice)"""

        """import matplotlib.pyplot as plt
        plt.plot(range(len(privacy_epsilons)), privacy_epsilons)
        plt.xlabel("Epochs")
        plt.ylabel("Privacy Budget (ε)")
        plt.title("Privacy Budget over Epochs")
        plt.show()"""
        model.train()
        # print("Best iteration: ", best_iter, "Best val dice: ", max_dice, "Best test dice: ", best_test_dice)
        # debugging learnable kernel
        if args.morphology and args.learnable_radius:
            with torch.no_grad():
                r = model.morph_layer.current_radius()
                radius_history.append(r)
                print(f"Epoch {t}: radius = {r:.3f}")

            # debugging learnable kernel
            model.train()
            images, labels = next(iter(train_loader))
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)  # this should pass through morph_layer
            loss = criterion_seg(out, labels.squeeze(1))
            loss.backward()
            print("grad of logit_radius:", model.morph_layer.logit_radius.grad)
            print("radius before step:", model.morph_layer.current_radius())
            optimizer.step()
            print("radius after step:", model.morph_layer.current_radius())



    #print(f" training loss:{training_losses}")
    #print(f" validation loss:{validation_losses}")
    if args.DPSGD==True:
        print(f" privacy_epsilons:{privacy_epsilons}")
        privacy_epsilons_str = str(privacy_epsilons)
    else:
        privacy_epsilons_str=0
    #print(f" validation dice score:{validation_dice_scores}")

    # privacy_epsilons_str = ""
    if not test:
        print("Best iteration: ", best_iter, "Best val dice: ", max_dice)

        if args.DPSGD== True:
            privacy_epsilon= privacy_epsilons[-1]
        else:
            privacy_epsilon=0


        file_name = "oct_results_UMN_Duke_lr_max_gradient.csv"
        wall_time_total_s = time.time() - t_total_start

        save_results_to_csv(args,model,
            file_name,
            max_grad_norm,
            noise_multiplier,
            model_name,
            learning_rate,
            batch_size,
            training_losses,
            validation_losses,
            dice_all_list,
            validation_dice_scores,
            privacy_epsilon,
            iterations,
            args.dataset,
            mae_all,
            per_layer_all_list_MAE,
            hd95_all,
            wall_time_total_s=wall_time_total_s,
            wall_time_train_s=t_train_acc,
            wall_time_val_s=t_val_acc,
            wall_time_test_s=t_test_acc,
            gpu_name=gpu_name,
            seed=args.seed,
            policy_type=args.policy_type,
            quantile_25_80=args.quantile_25_80,
            quantile_20_85=args.quantile_20_85,

        )
        print("saving has finished")
        #segmentation_plots(val_loader, model, device, model_name, args.DPSGD, args.dataset,num_examples=20 )


    if test:
        print("Test on Test data")
        print(f"if morphology was active:{args.morphology}")
        t_test_start = time.time()
        if not args.morphology:
            eval_use_morph = False
        elif not args.conditional_morph:
            eval_use_morph = True
        else:
            eval_use_morph = bool(getattr(model, "use_morph", args.use_morph))
        dice_test, test_loss, dice_all_test,mae_test,per_layer_all_MAE_test,mean_hd95 =eval(args,model_name,test_loader, criterion_seg, model, dice_s=True, n_classes=n_classes,use_morph=eval_use_morph,epoch=t, mode="test")
        t_test_acc += (time.time() - t_test_start)
        if args.DPSGD == True:
            privacy_epsilon = privacy_epsilons[-1]
        else:
            privacy_epsilon = 0 # None

        file_name = "test_oct_results_UMN_Duke_lr_max_gradient.csv"
        wall_time_total_s = time.time() - t_total_start
        save_results_to_csv(
            args,
            model,
            file_name,
            max_grad_norm,
            noise_multiplier,
            model_name,
            learning_rate,
            batch_size,
            training_losses,
            [test_loss],
            [dice_all_test],  # if you want a list,
            [dice_test],
            privacy_epsilon,
            iterations,
            args.dataset,
            [mae_test],  # keep the interface consistent with train case
            [per_layer_all_MAE_test],
            hd95_all,
            wall_time_total_s=wall_time_total_s,
            wall_time_train_s=t_train_acc,
            wall_time_val_s=t_val_acc,
            wall_time_test_s=t_test_acc,
            gpu_name=gpu_name,
            seed=args.seed,
            policy_type=args.policy_type,
            quantile_25_80=args.quantile_25_80,
            quantile_20_85=args.quantile_20_85,
        )
       # print(f"test losses:{test_loss}")
       # print(f"dice test:{dice_test}")
       #print(f"mae test:{mae_test}")

        # plotting
        #segmentation_plots(test_loader, model, device, model_name,args.DPSGD,args.dataset,num_examples=20)
    print(f" training loss:{training_losses}")
    print(f" validation loss:{validation_losses}")
    print(f"Validation HD95 (layers 1-7): {mean_hd95:.4f}")

    if args.morphology and args.learnable_radius:
        print(f"radius history is :{radius_history}")
        os.makedirs("plots", exist_ok=True)  # optional: folder for plots
        plt.figure()
        plt.plot(radius_history, training_losses, marker="o",label="radius")
        plt.xlabel("Effective radius")
        plt.ylabel("training_loss")
        #plt.title("Evolution of morphology radius")
        plt.grid(True)
        plt.savefig("plots/radius_history_vs_validation.pdf", dpi=300, bbox_inches="tight")  # PNG
    #plt.show()

    if args.collect_activation:
        for h in handles:
            h.remove()
        # saving the activations
        """eps_str = ""
        if args.DPSGD and len(privacy_epsilons) > 0:
            eps_str = f"{privacy_epsilons[-1]:.2f}"""

        # Aggregate per layer & per epoch
        agg_activations = {}  # layer_name -> { epoch -> np.array[C] }

        for layer_name, layer_epoch_dict in activations.items():
            layer_agg = {}
            for e, vec_list in layer_epoch_dict.items(): # here we have a vector for all batches for all channels for specific epoch number
                stacked = torch.stack(vec_list, dim=0)  # [num_batches, C] # stack it : make a 2 matrix : for example batches are rows and channels are columns
                mean_vec = stacked.mean(dim=0)  # [C] # average over batches for each channel (average of channel)
                layer_agg[e] = mean_vec.numpy() # e is epoch number
            agg_activations[layer_name] = layer_agg # the epoch info is inside layer_agg.
        # a view of agg_activations
        """agg_activations = {
        "0": {                   # <-- this is .keys() at top level
            10: np.array([...]), # epoch 10, shape [C]
            50: np.array([...]), # epoch 50, shape [C]
        },
        "3": {
            10: np.array([...]),
            50: np.array([...]),
        },
        }"""
        # Save everything to a file:
        # Save one file per layer

        for layer_name, layer_epoch_dict in agg_activations.items():
            save_path = os.path.join(
                save_dir_activations,
                f"{args.dataset}_activations_modelX_layer_{layer_name}_{model_name}_{eps_str}.pt"
            )
            torch.save(
                {
                    "layer_name": layer_name,
                    "epochs": sorted(list(layer_epoch_dict.keys())),
                    "activations": layer_epoch_dict,  # epoch -> np.array[C]
                },
                save_path,
            )
            print(f"Saved epoch activations for layer {layer_name} to: {save_path}")
    if args.DPSGD and len(gradient_cosines) > 0:
        grad_dir = "gradient_alignment_logs"
        os.makedirs(grad_dir, exist_ok=True)

        grad_df = pd.DataFrame({
            "epoch": gradient_cosine_epochs,
            "batch_idx": gradient_cosine_batches,
            "dataset": args.dataset,
            "model_name": args.model_name,
            "clipping": args.clipping,
            "max_grad_norm": args.max_grad_norm,
            "cosine": gradient_cosines,
            "seed": args.seed,
            "excluding_3_4_layer":args.excluding_3_4_layer
        })

        grad_path = os.path.join(
            grad_dir,
            f"{args.dataset}_{args.model_name}_{args.clipping}_eps{epsilon_val}_seed{args.seed}_morphology{args.morphology}_retinal_layer_wise_{args.retinal_layer_wise}_excl_three_four_{args.excluding_3_4_layer}.csv"
        )
        grad_df.to_csv(grad_path, index=False)
        print(f"Saved gradient alignment log to: {grad_path}")
    return model



import time
if __name__ == "__main__":
    # global step as a mutable container so hook can access it
    global activations, global_step
    global_step = {"value": 0}
    activations = {}
    args = argument_parser()
    start_time = time.time()
    print(args)
    print(f"Using seed: {args.seed}")
    set_seed(args.seed)
    #set_seed(args.seed)
    train(args)
    end_time = time.time()  # Record the end time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


# scp /home/parsar0000/pythonProject4/training.py shiva.parsarad@unibas.ch@chinchilla.dmi.unibas.ch:/home/parsar0000/BPR/pythonProject4




