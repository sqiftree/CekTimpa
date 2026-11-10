"""Bilateral spatial filter."""
import numpy as np
import cv2

def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 3,
    sigma_color: float = 25.5,
    sigma_space: float = 1.0,
) -> np.ndarray:
    """
    Menerapkan Bilateral Filtering (Kaplan dkk., 2020).
    Filter bilateral mempertimbangkan jarak spasial (sigma_space) dan perbedaan
    intensitas spektral (sigma_color), sehingga menghaluskan derau pada area
    homogen namun tetap mempertahankan ketajaman tepi batas objek.
    """
    return cv2.bilateralFilter(
        image,
        d=int(d),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )
