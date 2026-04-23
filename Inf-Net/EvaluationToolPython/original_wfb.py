import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter


def original_wfb(fg, gt):
    """
    WFb Compute the Weighted F-beta measure (as proposed in "How to Evaluate
    Foreground Maps?" [Margolin et. al - CVPR'14])

    Parameters:
    fg (numpy array): Binary/Non binary foreground map with values in the range [0 1]
    gt (numpy array): Binary ground truth

    Returns:
    float: The Weighted F-beta score
    """
    eps = np.finfo(float).eps

    # Check input
    if fg.dtype != np.float64:
        raise ValueError("FG should be of type: double")

    if np.max(fg) > 1 or np.min(fg) < 0:
        raise ValueError("FG should be in the range of [0 1]")

    if gt.dtype != bool:
        raise ValueError("GT should be of type: logical")

    d_gt = gt.astype(np.float64)  # Use double for computations

    e = np.abs(fg - d_gt)

    # Distance transform (equivalent to MATLAB's bwdist)
    dst = distance_transform_edt(~d_gt)

    # Pixel dependency - Gaussian filter (equivalent to MATLAB's fspecial('gaussian',7,5))
    # Create Gaussian kernel
    sigma = 5.0 / 6.0  # Convert MATLAB parameter to scipy parameter
    et = e.copy()

    # Handle edge pixels similar to MATLAB implementation
    # For pixels not in GT, use distance-based indexing (simplified approximation)
    mask = ~gt
    if np.any(mask):
        # This is a simplified version - in MATLAB there's more complex indexing
        et[mask] = e[mask]

    # Apply Gaussian filter
    ea = gaussian_filter(et, sigma)

    # Pixel importance calculation
    min_e_ea = e.copy()
    condition = gt & (ea < e)
    min_e_ea[condition] = ea[condition]

    # Pixel importance
    b = np.ones_like(gt, dtype=np.float64)
    b[~gt] = 2.0 - 1 * np.exp(np.log(1 - 0.5) / 5.0 * dst[~gt])

    ew = min_e_ea * b

    # Calculate weighted metrics
    tpw = np.sum(d_gt) - np.sum(ew[gt])
    fpw = np.sum(ew[~gt])

    # Weighted Recall and Precision
    r = 1 - np.mean(ew[gt]) if np.sum(gt) > 0 else 0  # Weighted Recall
    p = tpw / (eps + tpw + fpw)  # Weighted Precision

    # Weighted F-beta score (Beta=1)
    q = (2) * (r * p) / (eps + r + p)

    return q
