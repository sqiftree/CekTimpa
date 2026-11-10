"""Shared binary classification metrics."""
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
