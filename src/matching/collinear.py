"""Spatial shape filter for displacement clusters."""
from typing import List, Tuple
import numpy as np
def _filter_collinear_cluster(
    pair_indices: List[Tuple[int, int]],
    coords: np.ndarray,
    min_minor_axis: float = 2.5
) -> bool:
    """
    Memeriksa apakah kumpulan blok yang cocok membentuk gumpalan objek 2D (seperti tubuh orang)
    atau hanya berjajar membentuk garis lurus 1D (seperti lis atap gedung / zebra cross).
    Menggunakan analisis nilai eigen matriks kovariansi koordinat (PCA Spasial 2D).
    """
    if len(pair_indices) < 4:
        return False

    pts = np.array([coords[i] for i, _ in pair_indices], dtype=np.float32)
    if len(pts) < 4:
        return False

    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    if cov.ndim < 2:
        return False

    eigvals = np.linalg.eigvalsh(cov)
    minor_std = float(np.sqrt(max(eigvals[0], 0.0)))
    # Jika sebaran sumbu pendek (minor axis) terlalu tipis, berarti blok hanya berbaris di garis 1D
    return minor_std >= min_minor_axis
