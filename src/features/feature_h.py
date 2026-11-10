"""Intensity statistics feature (H)."""
import numpy as np

def extract_feature_h(blocks: np.ndarray) -> np.ndarray:
    """
    Fitur E: Statistik intensitas blok (Mean, Variance, Skewness).
    """
    blocks_f = blocks.astype(np.float32)
    mean = blocks_f.mean(axis=(1, 2))
    var = blocks_f.var(axis=(1, 2))
    std = np.sqrt(var) + 1e-6
    diff = blocks_f - mean[:, None, None]
    skew = np.mean(diff ** 3, axis=(1, 2)) / (std ** 3)

    return np.column_stack([mean, var, skew]).astype(np.float32)
