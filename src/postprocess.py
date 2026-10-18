"""
Modul Pasca-Pemrosesan Morfologi Citra
Materi PCD:
- Operasi Morfologi Biner: Erosi, Dilasi, Opening, Closing
- Hole Filling (Pengisian Lubang)
- Boundary Extraction: beta(A) = A - (A (-) SE)
"""

from typing import Tuple
import numpy as np
import cv2

def build_detection_mask(
    image_shape: Tuple[int, int],
    matched_pairs: list,
    coords: np.ndarray,
    block_size: int = 16,
    area_min: int = 100
) -> np.ndarray:
    """
    Membangun mask biner dari pasangan blok yang terdeteksi,
    dibersihkan dengan closing, opening, dan hole filling.
    """
    # TODO: Implementasi oleh mahasiswa sesuai materi morfologi
    raise NotImplementedError("Silakan implementasikan build_detection_mask")

def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Mengekstrak garis tepi area terdeteksi dengan rumus:
    beta(A) = A - (A (-) SE)
    """
    # TODO: Implementasi oleh mahasiswa
    raise NotImplementedError("Silakan implementasikan extract_boundary")
