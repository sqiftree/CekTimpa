"""Radial Fourier feature (D)."""
import numpy as np

def extract_feature_d(blocks: np.ndarray, radial_bins: int = None, normalize: bool = False) -> np.ndarray:
    """
    Fitur D: DFT Radial R(r) berdasar properti rotasi Fourier.
    """
    n, b, _ = blocks.shape
    fft_shifted = np.fft.fftshift(np.fft.fft2(blocks.astype(np.float32)), axes=(1, 2))
    mag = np.abs(fft_shifted)

    center = b // 2
    y, x = np.ogrid[:b, :b]
    r_map = np.sqrt((x - center)**2 + (y - center)**2).astype(np.int32)

    max_r = b // 2 if radial_bins is None else int(radial_bins)
    if not 1 <= max_r <= b // 2:
        raise ValueError("radial_bins must be between 1 and B//2")
    features = np.zeros((n, max_r), dtype=np.float32)
    for r in range(1, max_r + 1):
        mask = (r_map == r)
        if np.any(mask):
            features[:, r - 1] = mag[:, mask].mean(axis=1)

    if normalize:
        features /= b ** 2
    return features
