import numpy as np
from s_object import s_object
from s_region import s_region


def structure_measure(prediction, gt):
    """
    StructureMeasure computes the similarity between the foreground map and
    ground truth (as proposed in "Structure-measure: A new way to evaluate
    foreground maps" [Deng-Ping Fan et. al - ICCV 2017])

    Parameters:
    prediction (numpy array): Binary/Non binary foreground map with values in the range [0 1]
    gt (numpy array): Binary ground truth

    Returns:
    float: The computed similarity score
    """
    # Check input
    if not isinstance(prediction, np.ndarray) or prediction.dtype != np.float64:
        prediction = prediction.astype(np.float64)

    if np.max(prediction) > 1 or np.min(prediction) < 0:
        raise ValueError("The prediction should be in the range of [0 1]")

    if not isinstance(gt, np.ndarray) or gt.dtype != bool:
        gt = gt.astype(bool)

    y = np.mean(gt)

    if y == 0:  # if the GT is completely black
        x = np.mean(prediction)
        q = 1.0 - x  # only calculate the area of intersection
    elif y == 1:  # if the GT is completely white
        x = np.mean(prediction)
        q = x  # only calculate the area of intersection
    else:
        alpha = 0.5
        q = alpha * s_object(prediction, gt) + (1 - alpha) * s_region(prediction, gt)
        if q < 0:
            q = 0

    return q
