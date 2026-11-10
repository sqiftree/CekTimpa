"""Low-frequency zigzag DCT feature."""
import numpy as np
from scipy.fft import dctn

def extract_feature_dct(blocks: np.ndarray, n_coeffs: int = 16, include_dc: bool = False) -> np.ndarray:
    """Orthonormal DCT-II, low-frequency zigzag, excluding DC.

    Divide by block width to keep coefficient scale comparable across sizes.
    DC is omitted for invariance to additive brightness (without clipping).
    """
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
        raise ValueError("blocks must have shape (N, B, B)")
    b = blocks.shape[1]
    if not 1 <= n_coeffs <= b * b - (not include_dc):
        raise ValueError("n_coeffs must be between 1 and B*B-1")
    zigzag = []
    for diagonal in range(2 * b - 1):
        cells = [(y, diagonal - y) for y in range(b) if 0 <= diagonal - y < b]
        zigzag.extend(cells[::-1] if diagonal % 2 == 0 else cells)
    start = 0 if include_dc else 1
    selected = zigzag[start:start + n_coeffs]
    coefficients = dctn(blocks.astype(np.float32), axes=(1, 2), norm="ortho")
    ys, xs = zip(*selected)
    return (coefficients[:, ys, xs] / b).astype(np.float32)
