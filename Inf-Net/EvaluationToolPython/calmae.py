import numpy as np


def cal_mae(smap, gt_img):
    """
    Calculate the Mean Absolute Error (MAE) between saliency map and ground truth.

    This is a Python implementation of the MATLAB CalMAE function.

    Parameters:
    smap (numpy array): Saliency map with values in range [0, 1]
    gt_img (numpy array): Ground truth binary image

    Returns:
    float: The Mean Absolute Error
    """
    # Check if dimensions match
    if smap.shape[:2] != gt_img.shape[:2]:
        raise ValueError("Saliency map and gt Image have different sizes!")

    # Convert gt to logical if necessary
    if gt_img.dtype != bool:
        if len(gt_img.shape) == 3:
            gt_img = gt_img[:, :, 0] > 128
        else:
            gt_img = gt_img > 128

    # Convert smap to double and get first channel if needed
    if len(smap.shape) == 3:
        smap = smap[:, :, 0].astype(np.float64)
    else:
        smap = smap.astype(np.float64)

    # Calculate MAE using the original MATLAB logic
    fg_pixels = smap[gt_img]
    fg_err_sum = len(fg_pixels) - np.sum(fg_pixels)
    bg_err_sum = np.sum(smap[~gt_img])
    mae = (fg_err_sum + bg_err_sum) / gt_img.size

    return mae
