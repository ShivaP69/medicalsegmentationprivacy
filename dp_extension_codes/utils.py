#!/usr/bin/python
#
# Copyright 2022 Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import traceback

SEG_LABELS_LIST = [
    {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    {"id": 0, "name": "Region above the retina (RaR)", "rgb_values": [128, 0, 0]},
    {"id": 1, "name": "ILM: Inner limiting membrane", "rgb_values": [0, 128, 0]},
    {"id": 2, "name": "NFL-IPL: Nerve fiber ending to Inner plexiform layer", "rgb_values": [128, 128, 0]},
    {"id": 3, "name": "INL: Inner Nuclear layer", "rgb_values": [0, 0, 128]},
    {"id": 4, "name": "OPL: Outer plexiform layer", "rgb_values": [128, 0, 128]},
    {"id": 5, "name": "ONL-ISM: Outer Nuclear layer to Inner segment myeloid", "rgb_values": [0, 128, 128]},
    {"id": 6, "name": "ISE: Inner segment ellipsoid", "rgb_values": [128, 128, 128]},
    {"id": 7, "name": "OS-RPE: Outer segment to Retinal pigment epithelium", "rgb_values": [64, 0, 0]},
    {"id": 8, "name": "Region below RPE (RbR)", "rgb_values": [192, 0, 0]}]


def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1, 2, 0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


def plot_mult(labels, names, save=False, idx=0):
    n_c = len(labels)
    fig, axs = plt.subplots(nrows=1, ncols=n_c, figsize=(n_c*4,4))
    for i, ax in enumerate(axs.flatten()):
        ax.axis('off')
        plt.sca(ax)
        plt.imshow(labels[i])
        #plt.title(names[i])
    if save:
        plt.savefig("./figs/"+ str(idx) + ".png", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def mIOU(label, pred, num_classes=9):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)


"""def per_class_dice(y_pred, y_true, num_class):
    avg_dice = 0
    y_pred = y_pred.data.squeeze() #.cpu().numpy()

    y_true = y_true.data.squeeze() #.cpu().numpy()
    dice_all = np.zeros(num_class)
    for i in range(num_class):
        GT = y_true[:,:,i].view(-1)
        Pred = y_pred[:,:,i].view(-1)
        #print(GT.shape, Pred.shape)
        inter = (GT * Pred).sum() + 0.0001
        union = GT.sum()  + Pred.sum()  + 0.0001
        t = 2 * inter / union
        avg_dice = avg_dice + (t / num_class)
        dice_all[i] = t
    return avg_dice, dice_all
"""

"""def per_class_dice(y_pred, y_true, num_class):
    avg_dice = 0

    # for channel controlling:when number of channel is one and this one does not represented in the output of the network
    batch_size = y_true.shape[0]
    if batch_size !=1 :
        y_pred = y_pred.data.squeeze()  # .cpu().numpy(
        y_true = y_true.data.squeeze()  # .cpu().numpy()
    else:
        y_true = y_true[0,:,:,:]

    dice_all = np.zeros(num_class)

    for j in range(batch_size):
        image_true = y_true[j]
        image_pred = y_pred[j]
        #print(image_true.shape, image_pred.shape)
        image_pred = image_pred.reshape((image_pred.shape[0] * image_pred.shape[1], -1))
        image_true = image_true.reshape((image_true.shape[0] * image_true.shape[1], -1))
        for i in range(num_class):
            GT = image_true[:, i].view(-1)
            Pred = image_pred[:, i].view(-1)
            inter = (GT * Pred).sum() + 0.0001
            union = GT.sum() + Pred.sum() + 0.0001
            t = 2 * inter / union
            avg_dice = avg_dice + t
            dice_all[i] +=t
    dice_all = dice_all / batch_size
    avg_dice = avg_dice / batch_size
    print(f"avg_dice: {avg_dice}")
    return avg_dice, dice_all"""



def per_class_dice(y_pred, y_true, num_class):
    avg_dice = 0

    # for channel controlling:when number of channel is one and this one does not represented in the output of the network
    batch_size = y_true.shape[0]
    if batch_size !=1 :
        y_pred = y_pred.data.squeeze()  # .cpu().numpy(
        y_true = y_true.data.squeeze()  # .cpu().numpy()
    else:
        y_true = y_true[0,:,:,:]

    dice_all = np.zeros(num_class)

    for j in range(batch_size):
        image_true = y_true[j]
        image_pred = y_pred[j]

        if num_class>1:
            #print(image_true.shape, image_pred.shape)
            image_pred = image_pred.reshape((image_pred.shape[0] * image_pred.shape[1], -1))
            image_true = image_true.reshape((image_true.shape[0] * image_true.shape[1], -1))

            for i in range(num_class):
                GT = image_true[:, i].view(-1)
                Pred = image_pred[:, i].view(-1)
                #print(f"GT:{GT}")
                #print(f"Pred:{Pred}")
                inter = (GT * Pred).sum() #+ 0.0001
                #print(f"inter: {inter}")
                union = GT.sum() + Pred.sum() #+ 0.0001 # this extra value makes the calculation incorrect
                #print(f"union: {union}")
                if union == 0: # prevent division by zero
                    dice = torch.tensor(1.0, device=Pred.device)  # perfect match when nothing exists
                else:
                    dice = 2 * inter / union
                #print(f"dice: {dice}")
                avg_dice = avg_dice + (dice / num_class)
                dice_all[i] += dice
        else:
            image_pred = image_pred.reshape(-1)
            image_true=image_true.reshape(-1)
            inter  = torch.sum(image_true * image_pred) #+ 0.0001
            union = image_true.sum() + image_pred.sum() #+ 0.0001
            if union == 0:  # prevent division by zero
                dice = torch.tensor(1.0, device=image_pred.device)  # perfect match when nothing exists
            else:
                dice = 2 * inter / union
            # what we need here is calculating average dice and dice_all over all classes but here we have just 1 class, so there is no need for calculate average
            avg_dice+=dice
            dice_all[0]+=dice
    #print(f"dice_all before division: {dice_all}")
    #print(f"batch_size: {batch_size}")
    dice_all = dice_all / batch_size
    avg_dice = avg_dice / batch_size
    #print(f"avg_dice: {avg_dice}")
    #print(f"dice_all: {dice_all}")
    return avg_dice, dice_all




def compute_dice(ground_truth, prediction):

    #print("ground_truth", ground_truth.shape)
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        ret = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        for i in range(9):
            mask1 = (ground_truth == i)
            mask2 = (prediction == i)
            if mask1.sum() != 0:
                #print(mask1.shape)
                #print(ground_truth.shape)
                #print(prediction.shape)
                #print(ground_truth==prediction)
                ret[i] = float(2 * ((mask1 * (ground_truth == prediction)).sum()) / (mask1.sum() + mask2.sum()))
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret

def compute_pa(ground_truth, prediction):
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        ret = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        for i in range(9):
            mask1 = (ground_truth == i)
            if mask1.sum() != 0:
                ret[i] = float(((mask1 * (ground_truth == prediction)).sum()) / (mask1.sum()))
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret






def categorical_mae(y_pred, y_true):
    """
    Calculate the 'Mean Absolute Error' per class for categorical class predictions
    where y_pred is softmax probabilities and y_true is one-hot encoded.

    Args:
    y_pred (torch.Tensor): Softmax output of the model [batch, num_classes, ...].
    y_true (torch.Tensor): One-hot encoded true labels [batch, num_classes, ...].

    Returns:
    torch.Tensor: MAE per class, interpreted as misclassification rate.
    """
    # Convert softmax probabilities to predicted labels
    y_pred_labels = torch.argmax(y_pred, dim=1)

    # Convert one-hot encoded labels to single integer labels for comparison
    y_true_labels = torch.argmax(y_true, dim=1)

    # Calculate mismatches
    errors = y_pred_labels != y_true_labels

    # Calculate misclassification rate per class
    class_mae = []
    for i in range(y_true.size(1)):  # iterating over each class
        class_mask = y_true_labels == i
        if class_mask.any():
            class_errors = errors[class_mask]
            class_mae.append(class_errors.float().mean())  # Mean of errors in the class
        else:
            class_mae.append(torch.tensor(0.0))  # Avoid division by zero if no elements of this class

    class_mae = torch.stack(class_mae)

    return class_mae







def MAE(y_true, pred, n_classes):

    nc = pred.size()[1]
    pred = F.softmax(pred, dim=1)
    mae_sum=0
    per_layer=np.zeros(nc)
    for c in range(nc-1, 0, -1):
        pred_line = pred[:, c, :, :]
        gt_line  = (y_true == c).float()
        #print(f"pred_line: {pred_line.shape}")
        #print(f"gt_line: {gt_line.shape}")
        gt_line=gt_line.squeeze()
        mae_layer = torch.abs(gt_line - pred_line) # absolute error of layer c
        per_layer[c]= (torch.mean(mae_layer)).item()
        mae_sum+= torch.mean(mae_layer)
    return mae_sum/nc, per_layer


def MAE_New(y, p, n_classes,classes=None): # classes can be for example : [1,2,3,4,5,6,7,...]
    p = F.softmax(p, dim=1)
    #print(y.shape)
    #print(p.shape)
    B, C, H, W = y.shape
    assert C == n_classes
    maes = []
    mae_sum = 0
    per_layer = np.zeros(n_classes)
    for i in range(n_classes):
        t = y[:, i, :, :]
        r = p[:, i, :, :]
        m = torch.abs(t - r).mean()
        per_layer[i] = m.item()
        #mae_sum += m
        maes.append(m)
    if classes:
        mean_mae = torch.stack([maes[i] for i in classes]).mean()
    else:
        mean_mae = torch.stack(maes).mean()
    return mean_mae, per_layer

def MAE_secondversion(ground_truth, prediction):
    """
    Calculates Mean Absolute Error for multi-class segmentation with input shape [B, C, W, H].

    Args:
      ground_truth: Ground truth segmentation mask (tensor with shape [B, C, W, H]).
      prediction: Predicted segmentation mask (tensor with shape [B, C, W, H]).

    Returns:
      A dictionary containing the MAE for each class and the overall MAE.
    """
    prediction = F.softmax(prediction, dim=1)
    num_classes = ground_truth.shape[1]  # Number of classes is the second dimension
    maes = {}

    for class_idx in range(num_classes):
        gt_class = ground_truth[:, class_idx, ...]  # Select the current class for all batches
        pred_class = prediction[:, class_idx, ...]
        class_mae = torch.mean(torch.abs(gt_class - pred_class))
        maes[f"Class_{class_idx}"] = class_mae

    class_mae_tensor = torch.stack(list(maes.values()))
    overall_mae = torch.mean(class_mae_tensor)
    maes["Overall"] = overall_mae

    return maes


