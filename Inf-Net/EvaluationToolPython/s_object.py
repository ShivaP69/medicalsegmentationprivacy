import numpy as np


def s_object(prediction, gt):
    """
    S_object Computes the object similarity between foreground maps and ground
    truth (as proposed in "Structure-measure:A new way to evaluate foreground
    maps" [Deng-Ping Fan et. al - ICCV 2017])

    Parameters:
    prediction (numpy array): Binary/Non binary foreground map with values in the range [0 1]
    gt (numpy array): Binary ground truth

    Returns:
    float: The object similarity score
    """
    # compute the similarity of the foreground in the object level
    prediction_fg = prediction.copy()
    prediction_fg[~gt] = 0
    o_fg = object_measure(prediction_fg, gt)

    # compute the similarity of the background
    prediction_bg = 1.0 - prediction
    prediction_bg[gt] = 0
    o_bg = object_measure(prediction_bg, ~gt)

    # combine the foreground measure and background measure together
    u = np.mean(gt)
    q = u * o_fg + (1 - u) * o_bg

    return q


def object_measure(prediction, gt):
    """
    Calculate object similarity score.

    Parameters:
    prediction (numpy array): Prediction map
    gt (numpy array): Ground truth mask

    Returns:
    float: Object similarity score
    """
    eps = np.finfo(float).eps

    # check the input
    if prediction.size == 0:
        return 0

    if prediction.dtype.kind == "i":  # if integer
        prediction = prediction.astype(np.float64)

    if prediction.dtype != np.float64:
        raise ValueError("prediction should be of type: double")

    if np.max(prediction) > 1 or np.min(prediction) < 0:
        raise ValueError("prediction should be in the range of [0 1]")

    if gt.dtype != bool:
        raise ValueError("GT should be of type: logical")

    # compute the mean of the foreground or background in prediction
    if np.sum(gt) == 0:
        return 0

    x = np.mean(prediction[gt])

    # compute the standard deviations of the foreground or background in prediction
    sigma_x = np.std(prediction[gt])

    score = 2.0 * x / (x**2 + 1.0 + sigma_x + eps)

    return score
