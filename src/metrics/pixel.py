"""Pixel-level mask metrics."""
import numpy as np
from .binary import _binary_metrics


def prepare_ground_truth_mask(mask, jpeg=False):
    """Validate lossless binary masks and threshold lossy JPEG mask artifacts."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("mask must be a grayscale image")
    if jpeg:
        return np.where(mask >= 128, 255, 0).astype(np.uint8)
    if not np.isin(mask, [0, 255]).all():
        raise ValueError("mask must be binary 0/255")
    return mask.astype(np.uint8, copy=False)


def calculate_pixel_metrics(pred_mask, gt_mask):
    """Evaluate source+target masks, or target-only masks, consistently."""
    return _binary_metrics(pred_mask, gt_mask)
