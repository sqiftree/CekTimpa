"""Per-image feature vector standardization."""
import numpy as np

def standardize_feature_vectors(features):
    """Per-image z-scores / sqrt(D): Euclidean distance is standardized RMS.

    No labels or dataset fit. Constant dimensions contribute zero; distributions
    still vary by image, so this is an optional experiment, not calibration.
    """
    values = np.asarray(features, dtype=np.float32)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.)
    return ((values - mean) / scale / np.sqrt(values.shape[1])).astype(np.float32)
