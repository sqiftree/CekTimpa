"""Binary evaluation; nonzero mask pixels are positive, zero division returns 0.

Empty/empty masks have F1 and IoU 0 (no positive evidence). Evaluate image
false alarms separately; report pixel macro averages on forged images.
"""
import numpy as np


def _divide(a, b):
    return float(a / b) if b else 0.0


def _binary_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.shape != truth.shape or prediction.size == 0:
        raise ValueError("prediction and truth must have equal, nonempty shapes")
    prediction, truth = prediction != 0, truth != 0
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, total=prediction.size,
                specificity=_divide(tn, tn + fp),
                precision=_divide(tp, tp + fp), recall=_divide(tp, tp + fn),
                f1=_divide(2 * tp, 2 * tp + fp + fn),
                iou=_divide(tp, tp + fp + fn))


def calculate_pixel_metrics(pred_mask, gt_mask):
    """Evaluate source+target masks, or target-only masks, consistently."""
    return _binary_metrics(pred_mask, gt_mask)


def calculate_image_metrics(predictions, true_labels):
    """Binary image labels. Balanced accuracy requires both classes."""
    predictions, true_labels = np.asarray(predictions), np.asarray(true_labels)
    if predictions.ndim != 1 or true_labels.ndim != 1:
        raise ValueError("image labels must be one-dimensional")
    if not np.isin(predictions, [0, 1]).all() or not np.isin(true_labels, [0, 1]).all():
        raise ValueError("image labels must be binary")
    m = _binary_metrics(predictions, true_labels)
    m['tpr'] = m['recall']
    m['fpr'] = _divide(m['fp'], m['fp'] + m['tn'])
    m['accuracy'] = _divide(m['tp'] + m['tn'], len(predictions))
    m['balanced_accuracy'] = ((m['tpr'] + 1 - m['fpr']) / 2
                              if m['tp'] + m['fn'] and m['tn'] + m['fp']
                              else float('nan'))
    return m


def calculate_psnr(image_a, image_b, data_range=255.0):
    """PSNR for 0..255 images; pass data_range=1 for normalized images."""
    a, b = np.asarray(image_a), np.asarray(image_b)
    if a.shape != b.shape or a.size == 0 or data_range <= 0:
        raise ValueError("equal nonempty shapes and positive data_range required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("images must contain finite values")
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))
