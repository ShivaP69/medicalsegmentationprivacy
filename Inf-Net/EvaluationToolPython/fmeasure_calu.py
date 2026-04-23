import numpy as np


def fmeasure_calu(s_map, gt_map, gt_size, threshold):
    """
    Calculate F-measure, precision, recall, specificity, and dice coefficient
    """
    if threshold > 1:
        threshold = 1

    # Ensure both arrays are boolean for bitwise operations
    label3 = (s_map >= threshold).astype(bool)
    gt_map_bool = gt_map.astype(bool)

    num_rec = np.sum(label3 == 1)  # FP + TP
    num_no_rec = np.sum(label3 == 0)  # FN + TN

    # Use boolean arrays for logical AND operation
    label_and = label3 & gt_map_bool
    num_and = np.sum(label_and)  # TP

    num_obj = np.sum(gt_map_bool)  # TP + FN
    num_pred = np.sum(label3)  # FP + TP

    fn = num_obj - num_and
    fp = num_rec - num_and
    tn = num_no_rec - fn

    if num_and == 0:
        pre_ftem = 0
        recall_ftem = 0
        fmeasure_f = 0
        dice = 0
        specif_tem = 0
    else:
        pre_ftem = num_and / num_rec if num_rec > 0 else 0
        recall_ftem = num_and / num_obj if num_obj > 0 else 0
        specif_tem = tn / (tn + fp) if (tn + fp) > 0 else 0
        dice = 2 * num_and / (num_obj + num_pred) if (num_obj + num_pred) > 0 else 0
        fmeasure_f = (
            (2.0 * pre_ftem * recall_ftem) / (pre_ftem + recall_ftem)
            if (pre_ftem + recall_ftem) > 0
            else 0
        )

    return pre_ftem, recall_ftem, specif_tem, dice, fmeasure_f
