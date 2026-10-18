"""
Modul Utilitas & Helper Visualisasi CekTimpa
"""

import numpy as np
import cv2

def create_overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Membuat visualisasi overlay warna transparan pada area terdeteksi."""
    # TODO: Implementasi helper overlay
    raise NotImplementedError()

def draw_shift_vectors(image_rgb: np.ndarray, matched_pairs: list, coords: np.ndarray) -> np.ndarray:
    """Menggambar garis penghubung antar pasangan blok yang cocok."""
    # TODO: Implementasi penggambaran vektor translasi
    raise NotImplementedError()
