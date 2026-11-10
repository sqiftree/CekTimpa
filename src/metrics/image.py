"""Image-level binary classification metrics."""
import numpy as np
from .binary import _binary_metrics, _divide
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
