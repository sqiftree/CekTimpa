"""Image degradation algorithm."""
import numpy as np
import cv2

def apply_gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Menerapkan penghalusan Gaussian blur dengan kernel ukuran ganjil otomatis (6*sigma + 1).
    """
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))
