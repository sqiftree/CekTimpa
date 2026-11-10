"""Multiscale bilateral detail decomposition."""
import numpy as np
import cv2
from .bilateral_filter import apply_bilateral_filter

def compute_bilateral_multiscale_detail(
    image: np.ndarray,
    d: int = 3,
    sigma_space: float = 1.0,
    sigma_color: float = 25.5,
) -> np.ndarray:
    """
    Menghitung dekomposisi multiskala bilateral untuk menghasilkan layer detail (D^2):
      Level 1: BF^1(I) menggunakan sigma_s dan sigma_r
      Level 2: BF^2(I) diterapkan pada BF^1 dengan menggandakan sigma_s (2 * sigma_s)
               dan membagi dua sigma_r (sigma_r / 2)
      Detail Layer: D^2(I) = BF^1(I) - BF^2(I)
    """
    bf1 = apply_bilateral_filter(
        image,
        d=d,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
    ).astype(np.float32)

    bf2 = cv2.bilateralFilter(
        bf1.astype(np.uint8) if bf1.dtype != np.uint8 else bf1,
        d=d,
        sigmaColor=float(sigma_color / 2.0),
        sigmaSpace=float(sigma_space * 2.0),
    ).astype(np.float32)

    detail_layer = bf1 - bf2
    return detail_layer
