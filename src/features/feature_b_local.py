"""Gaussian feature sampling with per-block context."""
import numpy as np
import cv2
def extract_feature_b_local(blocks, sigma=1.0, g_samples=4):
    """Gaussian smoothing inside each block; outside context cannot alter matches."""
    b = blocks.shape[1]
    if not np.isfinite(sigma) or sigma <= 0 or not 1 <= g_samples <= b:
        raise ValueError("positive sigma and 1 <= g_samples <= B required")
    samples = np.linspace(0, b - 1, g_samples, dtype=int)
    features = np.empty((len(blocks), g_samples ** 2), dtype=np.float32)
    for i, block in enumerate(blocks):
        blurred = cv2.GaussianBlur(block.astype(np.float32), (0, 0), sigma,
                                   borderType=cv2.BORDER_REFLECT_101)
        features[i] = blurred[np.ix_(samples, samples)].ravel()
    return features
