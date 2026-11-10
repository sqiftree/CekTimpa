"""Pixel-level mask metrics."""
from .binary import _binary_metrics
def calculate_pixel_metrics(pred_mask, gt_mask):
    """Evaluate source+target masks, or target-only masks, consistently."""
    return _binary_metrics(pred_mask, gt_mask)
