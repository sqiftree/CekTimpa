"""
Modul Preprocessing Citra (Pra-pemrosesan)
Materi PCD:
- Intensity Transformation & Color Models (RGB, Grayscale, HSI)
- Spatial Filtering (Box, Gaussian lowpass, Median)
"""

from typing import Tuple
import numpy as np
import cv2

def to_grayscale(image_rgb: np.ndarray, method: str = "intensity") -> np.ndarray:
    """
    Mengonversi citra RGB ke skala keabuan (grayscale).
    method:
      - 'intensity': I = (R + G + B) / 3 (konsisten dengan model HSI)
      - 'luminance': Standar ITU-R 601 (0.299 R + 0.587 G + 0.114 B)
    """
    if len(image_rgb.shape) == 2:
        return image_rgb.astype(np.uint8)

    img_float = image_rgb.astype(np.float32)
    if method == "intensity":
        gray = np.mean(img_float, axis=2)
    else:
        gray = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    
    return np.clip(gray, 0, 255).astype(np.uint8)

def rgb_to_hsi(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mengonversi citra RGB ke model warna HSI (Hue, Saturation, Intensity).
    Penanganan khusus pembagian nol saat R=G=B atau saturasi rendah.
    """
    img = image_rgb.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Intensitas I
    intensity = (r + g + b) / 3.0

    # Saturasi S
    min_rgb = np.minimum(np.minimum(r, g), b)
    sum_rgb = r + g + b
    saturation = np.zeros_like(intensity)
    non_zero = sum_rgb > 1e-6
    saturation[non_zero] = 1.0 - (3.0 / sum_rgb[non_zero]) * min_rgb[non_zero]

    # Hue H
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-7
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))

    hue = theta.copy()
    hue[b > g] = 2.0 * np.pi - hue[b > g]
    hue = hue / (2.0 * np.pi)  # Normalisasi ke [0, 1]

    return hue, saturation, intensity

def apply_prefilter(image: np.ndarray, filter_type: str = "none", **kwargs) -> np.ndarray:
    """
    Menerapkan pra-filter restorasi: 'none', 'median', atau 'gaussian'.
    """
    if filter_type == "median":
        ksize = kwargs.get("ksize", 3)
        return cv2.medianBlur(image, ksize)
    elif filter_type == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return image
