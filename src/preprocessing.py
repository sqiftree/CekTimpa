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
    method: 'intensity' (I = (R+G+B)/3 konsisten dengan HSI) atau 'luminance' (ITU-R 601).
    """
    # TODO: Implementasi oleh mahasiswa sesuai rumus kuliah
    raise NotImplementedError("Silakan implementasikan fungsi to_grayscale")

def rgb_to_hsi(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mengonversi citra RGB ke model warna HSI (Hue, Saturation, Intensity).
    Penanganan khusus pembagian dengan nol saat R=G=B atau saturasi rendah.
    """
    # TODO: Implementasi oleh mahasiswa sesuai rumus kuliah
    raise NotImplementedError("Silakan implementasikan fungsi rgb_to_hsi")

def apply_prefilter(image: np.ndarray, filter_type: str = "none", **kwargs) -> np.ndarray:
    """
    Menerapkan pra-filter restorasi: 'none', 'median' (kernel size), atau 'gaussian' (sigma).
    """
    # TODO: Implementasi oleh mahasiswa sesuai rumus kuliah
    raise NotImplementedError("Silakan implementasikan fungsi apply_prefilter")
