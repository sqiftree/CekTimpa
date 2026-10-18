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
    
    Args:
        image_channel: Citra 2D skala keabuan berdimensi (H, W)
        block_size: Panjang sisi blok persegi B (default: 16)
        stride: Jarak langkah pergeseran jendela s (default: 2)

    Returns:
        blocks: np.ndarray shape (N, B, B)
        coords: np.ndarray shape (N, 2) koordinat (y, x) pojok kiri atas tiap blok
    """
    h, w = image_channel.shape[:2]
    y_indices = np.arange(0, h - block_size + 1, stride, dtype=np.int32)
    x_indices = np.arange(0, w - block_size + 1, stride, dtype=np.int32)

    # Grid koordinat (y, x)
    grid_y, grid_x = np.meshgrid(y_indices, x_indices, indexing="ij")
    coords = np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)

    n_blocks = len(coords)
    blocks = np.empty((n_blocks, block_size, block_size), dtype=image_channel.dtype)
    
    for i in range(n_blocks):
        y, x = coords[i]
        blocks[i] = image_channel[y : y + block_size, x : x + block_size]

    return blocks, coords
