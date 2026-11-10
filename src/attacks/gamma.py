"""Image degradation algorithm."""
import numpy as np
import cv2

def apply_gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Menerapkan transformasi power-law (gamma correction):
    s = c * r^gamma
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    exponent = gamma
    table = np.array([((i / 255.0) ** exponent) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)
