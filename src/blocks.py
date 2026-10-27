"""
Modul Partisi & Ekstraksi Blok (Neighborhood Concept)
Materi PCD:
- Image Representation, Sampling & Neighborhood Operations
- Spatial Filtering & Variance Analysis (Penapisan Latar Belakang Polos / Langit)
- 2D Gradient & Structure Tensor Analysis (Fokus Objek 2D / Penekanan Garis 1D)
"""

from typing import Tuple
import numpy as np


def extract_blocks(
    image_channel: np.ndarray,
    block_size: int = 16,
    stride: int = 2,
    min_std: float = 0.0,
    min_grad_2d: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mengekstrak blok-blok bertumpuk (overlapping blocks) ukuran B x B dengan langkah stride s,
    dilengkapi penyaringan variansi lokal (min_std) dan variasi gradien 2D (min_grad_2d).

    Args:
        image_channel: Citra 2D skala keabuan berdimensi (H, W).
        block_size: Panjang sisi blok persegi B (default: 16).
        stride: Jarak langkah pergeseran jendela s (default: 2).
        min_std: Ambang batas simpangan baku minimum untuk membuang area polos (langit/tembok).
        min_grad_2d: Ambang batas variasi gradien 2D minimum untuk menekan garis lurus 1D
                     dan memfokuskan deteksi pada objek bertekstur.

    Returns:
        blocks: np.ndarray shape (N, B, B)
        coords: np.ndarray shape (N, 2) koordinat (y, x) pojok kiri atas tiap blok
    """
    h, w = image_channel.shape[:2]
    if h < block_size or w < block_size:
        return (
            np.empty((0, block_size, block_size), dtype=image_channel.dtype),
            np.empty((0, 2), dtype=np.int32)
        )

    y_indices = np.arange(0, h - block_size + 1, stride, dtype=np.int32)
    x_indices = np.arange(0, w - block_size + 1, stride, dtype=np.int32)

    # Grid koordinat (y, x)
    grid_y, grid_x = np.meshgrid(y_indices, x_indices, indexing="ij")
    coords = np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)

    n_blocks = len(coords)
    if n_blocks == 0:
        return (
            np.empty((0, block_size, block_size), dtype=image_channel.dtype),
            np.empty((0, 2), dtype=np.int32)
        )

    # Ekstraksi blok secara efisien
    blocks = np.empty((n_blocks, block_size, block_size), dtype=image_channel.dtype)
    for i in range(n_blocks):
        y, x = coords[i]
        blocks[i] = image_channel[y : y + block_size, x : x + block_size]

    # Penyaringan blok jika min_std > 0 atau min_grad_2d > 0
    valid_mask = np.ones(n_blocks, dtype=bool)

    if min_std > 0.0:
        stds = np.std(blocks, axis=(1, 2))
        valid_mask &= (stds >= min_std)

    if min_grad_2d > 0.0 and np.any(valid_mask):
        # Hitung struktur gradien 2D (eigenvalue terkecil dari matriks struktur gradien)
        blocks_f = blocks.astype(np.float32)
        gy = np.diff(blocks_f, axis=1)[:, :, :-1]
        gx = np.diff(blocks_f, axis=2)[:, :-1, :]

        j_yy = np.mean(gy * gy, axis=(1, 2))
        j_xx = np.mean(gx * gx, axis=(1, 2))
        j_yx = np.mean(gy * gx, axis=(1, 2))

        trace = j_yy + j_xx
        disc = np.sqrt(np.maximum(0.0, (j_yy - j_xx) ** 2 + 4.0 * (j_yx ** 2)))
        lambda2 = 0.5 * (trace - disc)
        grad_2d = np.sqrt(np.maximum(0.0, lambda2))

        valid_mask &= (grad_2d >= min_grad_2d)

    if not np.all(valid_mask):
        blocks = blocks[valid_mask]
        coords = coords[valid_mask]

    return blocks, coords
