import numpy as np


def enhanced_measure(fm, gt):
    """
    Compute the Enhanced Alignment measure (E-measure).

    This is a Python implementation of the MATLAB Enhancedmeasure function.
    As proposed in "Enhanced-alignment Measure for Binary Foreground Map Evaluation"
    [Deng-Ping Fan et. al - IJCAI'18 oral paper]

    Parameters:
    fm (numpy array): Binary foreground map
    gt (numpy array): Binary ground truth

    Returns:
    float: The Enhanced alignment score
    """
    # Ensure inputs are boolean for logical operations
    fm = fm.astype(bool)
    gt = gt.astype(bool)

    # Use double for computations
    d_fm = fm.astype(np.float64)
    d_gt = gt.astype(np.float64)

    # Special cases
    if np.sum(d_gt) == 0:  # if the GT is completely black
        enhanced_matrix = 1.0 - d_fm  # only calculate the black area of intersection
    elif np.sum(d_gt) == np.prod(d_gt.shape):  # if the GT is completely white (all 1s)
        enhanced_matrix = d_fm  # only calculate the white area of intersection
    else:
        # Normal case:
        # 1. compute alignment matrix
        align_matrix = alignment_term(d_fm, d_gt)
        # 2. compute enhanced alignment matrix
        enhanced_matrix = enhanced_alignment_term(align_matrix)

    # 3. E-measure score
    w, h = gt.shape
    eps = np.finfo(float).eps
    score = np.sum(enhanced_matrix) / (w * h - 1 + eps)

    return score


def alignment_term(d_fm, d_gt):
    """
    Compute alignment matrix.

    Parameters:
    d_fm (numpy array): Double precision foreground map
    d_gt (numpy array): Double precision ground truth

    Returns:
    numpy array: Alignment matrix
    """
    eps = np.finfo(float).eps

    # compute global mean
    mu_fm = np.mean(d_fm)
    mu_gt = np.mean(d_gt)

    # compute the bias matrix
    align_fm = d_fm - mu_fm
    align_gt = d_gt - mu_gt

    # compute alignment matrix
    align_matrix = (
        2.0 * (align_gt * align_fm) / (align_gt * align_gt + align_fm * align_fm + eps)
    )

    return align_matrix


def enhanced_alignment_term(align_matrix):
    """
    Enhanced Alignment Term function. f(x) = 1/4*(1 + x)^2)

    Parameters:
    align_matrix (numpy array): Alignment matrix

    Returns:
    numpy array: Enhanced alignment matrix
    """
    enhanced = ((align_matrix + 1) ** 2) / 4
    return enhanced
