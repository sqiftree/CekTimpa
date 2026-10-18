"""
Modul Simulasi Gangguan Citra (Attacks / Degradasi)
Materi PCD:
- Noise Models: Gaussian Noise & Salt-and-Pepper
- Spatial Filtering: Gaussian Blur
- Intensity Transformation: Power-law (Gamma) & Contrast Stretching
- Histogram Equalization
- JPEG Compression (Lossy)
"""

from typing import Optional
import numpy as np
import cv2

def apply_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 10.0, seed: Optional[int] = None) -> np.ndarray:
    """Menambahkan derau Gaussian pada citra."""
    raise NotImplementedError()

def apply_salt_pepper_noise(image: np.ndarray, amount: float = 0.02, seed: Optional[int] = None) -> np.ndarray:
    """Menambahkan derau biner salt-and-pepper."""
    raise NotImplementedError()

def apply_gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Menerapkan penghalusan Gaussian blur."""
    raise NotImplementedError()

def apply_gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Menerapkan transformasi power-law (gamma)."""
    raise NotImplementedError()

def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Menerapkan perataan histogram (histogram equalization)."""
    raise NotImplementedError()

def apply_jpeg_compression(image: np.ndarray, quality: int = 75) -> np.ndarray:
    """Menerapkan kompresi JPEG dengan tingkat kualitas tertentu."""
    raise NotImplementedError()
