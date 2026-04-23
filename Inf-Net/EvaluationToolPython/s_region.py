import numpy as np


def s_region(prediction, gt):
    """
    S_region computes the region similarity between the foreground map and
    ground truth (as proposed in "Structure-measure:A new way to evaluate
    foreground maps" [Deng-Ping Fan et. al - ICCV 2017])

    Parameters:
    prediction (numpy array): Binary/Non binary foreground map with values in the range [0 1]
    gt (numpy array): Binary ground truth

    Returns:
    float: The region similarity score
    """
    # find the centroid of the GT
    x, y = centroid(gt)

    # divide GT into 4 regions
    gt_1, gt_2, gt_3, gt_4, w1, w2, w3, w4 = divide_gt(gt, x, y)

    # Divide prediction into 4 regions
    prediction_1, prediction_2, prediction_3, prediction_4 = divide_prediction(
        prediction, x, y
    )

    # Compute the ssim score for each regions
    q1 = ssim(prediction_1, gt_1)
    q2 = ssim(prediction_2, gt_2)
    q3 = ssim(prediction_3, gt_3)
    q4 = ssim(prediction_4, gt_4)

    # Sum the 4 scores
    q = w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4

    return q


def centroid(gt):
    """
    Centroid Compute the centroid of the GT

    Parameters:
    gt (numpy array): Binary ground truth

    Returns:
    tuple: (X, Y) - The coordinates of centroid
    """
    rows, cols = gt.shape

    if np.sum(gt) == 0:
        x = round(cols / 2)
        y = round(rows / 2)
    else:
        total = np.sum(gt)
        i = np.arange(1, cols + 1)
        j = np.arange(1, rows + 1).reshape(-1, 1)
        x = round(np.sum(np.sum(gt, axis=0) * i) / total)
        y = round(np.sum(np.sum(gt, axis=1) * j.flatten()) / total)

    return x, y


def divide_gt(gt, x, y):
    """
    Divide the GT into 4 regions according to the centroid of the GT and return the weights

    Parameters:
    gt (numpy array): Ground truth
    x, y (int): Centroid coordinates

    Returns:
    tuple: (LT, RT, LB, RB, w1, w2, w3, w4)
    """
    # width and height of the GT
    hei, wid = gt.shape
    area = wid * hei

    # copy the 4 regions
    # Convert to 0-based indexing for Python
    lt = gt[0:y, 0:x]
    rt = gt[0:y, x:wid]
    lb = gt[y:hei, 0:x]
    rb = gt[y:hei, x:wid]

    # The different weight (each block proportional to the GT foreground region)
    w1 = (x * y) / area
    w2 = ((wid - x) * y) / area
    w3 = (x * (hei - y)) / area
    w4 = 1.0 - w1 - w2 - w3

    return lt, rt, lb, rb, w1, w2, w3, w4


def divide_prediction(prediction, x, y):
    """
    Divide the prediction into 4 regions according to the centroid of the GT

    Parameters:
    prediction (numpy array): Prediction map
    x, y (int): Centroid coordinates

    Returns:
    tuple: (LT, RT, LB, RB)
    """
    # width and height of the prediction
    hei, wid = prediction.shape

    # copy the 4 regions
    # Convert to 0-based indexing for Python
    lt = prediction[0:y, 0:x]
    rt = prediction[0:y, x:wid]
    lb = prediction[y:hei, 0:x]
    rb = prediction[y:hei, x:wid]

    return lt, rt, lb, rb


def ssim(prediction, gt):
    """
    ssim computes the region similarity between foreground maps and ground
    truth (as proposed in "Structure-measure: A new way to evaluate foreground
    maps" [Deng-Ping Fan et. al - ICCV 2017])

    Parameters:
    prediction (numpy array): Binary/Non binary foreground map with values in the range [0 1]
    gt (numpy array): Binary ground truth

    Returns:
    float: The region similarity score
    """
    eps = np.finfo(float).eps

    d_gt = gt.astype(np.float64)

    hei, wid = prediction.shape
    n = wid * hei

    # Compute the mean of SM, GT
    x = np.mean(prediction)
    y = np.mean(d_gt)

    # Compute the variance of SM, GT
    sigma_x2 = np.sum((prediction - x) ** 2) / (n - 1 + eps)
    sigma_y2 = np.sum((d_gt - y) ** 2) / (n - 1 + eps)

    # Compute the covariance between SM and GT
    sigma_xy = np.sum((prediction - x) * (d_gt - y)) / (n - 1 + eps)

    alpha = 4 * x * y * sigma_xy
    beta = (x**2 + y**2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        q = alpha / (beta + eps)
    elif alpha == 0 and beta == 0:
        q = 1.0
    else:
        q = 0

    return q
