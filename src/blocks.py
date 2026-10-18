"""
Modul Partisi & Ekstraksi Blok (Neighborhood Concept)
Materi PCD:
- Image Representation, Sampling & Neighborhood Operations
"""

from typing import Tuple
import numpy as np

def extract_blocks(
    image_channel: np.ndarray,
    block_size: int = 16,
    stride: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mengekstrak blok-blok bertumpuk (overlapping blocks) ukuran B x B dengan langkah stride s.
    
    Returns:
        blocks: np.ndarray shape (N, B, B)
        coords: np.ndarray shape (N, 2) koordinat (y, x) pojok kiri atas tiap blok
    """
    # TODO: Implementasi oleh mahasiswa dengan sliding_window / as_strided NumPy
    raise NotImplementedError("Silakan implementasikan fungsi extract_blocks")
