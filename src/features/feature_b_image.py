"""Gaussian feature sampling using image-wide context."""
import numpy as np
import cv2
def extract_feature_b(
    image_gray: np.ndarray,
    coords: np.ndarray,
    block_size: int = 16,
    sigma: float = 1.0,
    g_samples: int = 4
) -> np.ndarray:
    """
    Fitur B: Filter Gaussian Lowpass + subsampling teratur di dalam blok.
    """
    if not np.isfinite(sigma) or sigma <= 0 or not 1 <= g_samples <= block_size:
        raise ValueError("positive sigma and 1 <= g_samples <= B required")
    blurred = cv2.GaussianBlur(image_gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    step = block_size // g_samples

    n_blocks = len(coords)
    features = np.empty((n_blocks, g_samples * g_samples), dtype=np.float32)

    for i in range(n_blocks):
        y, x = coords[i]
        patch = blurred[y : y + block_size : step, x : x + block_size : step]
        features[i] = patch[:g_samples, :g_samples].ravel()

    return features
